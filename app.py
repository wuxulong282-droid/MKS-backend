from flask import Flask, request, jsonify, send_from_directory, Response, make_response
from flask_cors import CORS
import base64
import json
import os
import sys

# Windows asyncio 修复（必须在最顶部）
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import (DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
                    MODEL_NAME, MARX_SYSTEM_PROMPT)
# ASR 模块（云端可缺少 numpy）
try:
    import asr
    _HAS_ASR = True
except ImportError:
    _HAS_ASR = False
    print("[警告] ASR 模块导入失败（numpy 缺失），语音输入不可用", file=sys.stderr)

import tts
import llm

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# ── 静态文件 ──
@app.route('/')
def index():
    resp = make_response(send_from_directory(FRONTEND_DIR, 'index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not found"}), 404

# ── 健康检查 ──
@app.route('/health')
def health_check():
    return jsonify({"status":"ok"})

def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "voice": "zh-CN-YunjianNeural"
    })

# ── 文字对话（非流式，返回文字+音频）──
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json or {}
        message = data.get('message', '').strip()
        history = data.get('history', [])
        need_tts = data.get('tts', False)

        if not message:
            return jsonify({"error": "消息不能为空"}), 400

        reply = ""
        for chunk in llm.chat_stream(
            message, history, MARX_SYSTEM_PROMPT,
            DEEPSEEK_API_KEY, DEEPSEEK_API_URL, MODEL_NAME
        ):
            reply += chunk

        audio = tts.generate(reply) if need_tts else None
        return jsonify({"reply": reply, "audio": audio, "error": None})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"reply": "出错了，请重试", "audio": None, "error": None}), 200

# ── 文字对话（流式，Server-Sent Events）──
@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    try:
        data = request.json or {}
        message = data.get('message', '').strip()
        history = data.get('history', [])
        if not message:
            return jsonify({"error": "消息不能为空"}), 400

        def generate():
            import time
            sentence_buf = ""
            full_reply = ""
            first_sentence = True
            llm_start = time.time()
            for delta in llm.chat_stream(
                message, history, MARX_SYSTEM_PROMPT,
                DEEPSEEK_API_KEY, DEEPSEEK_API_URL, MODEL_NAME
            ):
                sentence_buf += delta
                full_reply += delta
                yield f"data: {json.dumps({'type': 'text', 'delta': delta})}\n\n"

                if any(c in delta for c in "。！？…\n"):
                    sentence = sentence_buf.strip()
                    sentence_buf = ""
                    if sentence:
                        if first_sentence:
                            first_sentence = False
                            print(f"[延迟] 第一句LLM完成 耗时{time.time()-llm_start:.2f}s")
                        tts_t0 = time.time()
                        audio = tts.generate(sentence)
                        if audio:
                            print(f"[延迟] TTS完成 耗时{time.time()-tts_t0:.2f}s")
                            yield f"data: {json.dumps({'type': 'audio', 'audio': audio})}\n\n"

            if sentence_buf.strip():
                tts_t0 = time.time()
                audio = tts.generate(sentence_buf.strip())
                if audio:
                    print(f"[延迟] 剩余TTS完成 耗时{time.time()-tts_t0:.2f}s")
                    yield f"data: {json.dumps({'type': 'audio', 'audio': audio})}\n\n"

            print(f"[延迟] 总耗时{time.time()-llm_start:.2f}s {len(full_reply)}字")
            yield f"data: {json.dumps({'type': 'done', 'full_text': full_reply})}\n\n"

        return Response(generate(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── 语音消息（按住说话→识别→回复）──
@app.route('/chat_voice', methods=['POST'])
def chat_voice():
    print('[voice] 收到请求')
    try:
        import time
        t0 = time.time()
        data = request.json or {}
        audio_b64 = data.get('audio_b64', '') or data.get('audio', '')
        history = data.get('history', [])

        if not audio_b64:
            return jsonify({"reply": "没有收到音频", "audio": None}), 200

        audio_bytes = base64.b64decode(audio_b64)
        print(f'[voice] 音频大小: {len(audio_bytes)} bytes')

        print('[voice] 开始识别...')
        if _HAS_ASR:
            text = asr.transcribe(audio_bytes)
        else:
            print("[voice] ASR 不可用，模拟空识别")
            text = ""
        print(f'[voice] 识别结果: "{text}"')
        if not text:
            print('[voice] 识别为空，返回提示')
            return jsonify({"reply": "", "audio": None}), 200
        print(f'[voice] 发送给LLM: {text[:30]}')
        print(f"[延迟] 按住说话:ASR完成 耗时{time.time()-t0:.2f}s 识别:{text[:30]}")

        t1 = time.time()
        reply = ""
        for chunk in llm.chat_stream(
            text, history, MARX_SYSTEM_PROMPT,
            DEEPSEEK_API_KEY, DEEPSEEK_API_URL, MODEL_NAME
        ):
            reply += chunk
        print(f"[延迟] 按住说话:LLM完成 耗时{time.time()-t1:.2f}s 回复{len(reply)}字")

        t2 = time.time()
        audio = tts.generate(reply)
        print(f"[延迟] 按住说话:TTS完成 耗时{time.time()-t2:.2f}s")
        print(f"[延迟] 按住说话:总耗时{time.time()-t0:.2f}s")

        return jsonify({
            "transcript": text,
            "reply": reply,
            "audio": audio,
            "error": None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[voice] 异常: {e}')
        return jsonify({"reply": "处理出错，请重试", "audio": None, "error": None}), 200

# ── 语音识别（仅转文字，不含LLM）──
@app.route('/asr', methods=['POST'])
def asr_route():
    try:
        import time
        t0 = time.time()
        data = request.json or {}
        audio_b64 = data.get('audio_b64', '') or data.get('audio', '')
        if not audio_b64:
            return jsonify({"text": "", "error": None}), 200
        audio_bytes = base64.b64decode(audio_b64)
        if _HAS_ASR:
            text = asr.transcribe(audio_bytes)
        else:
            print("[asr] ASR 不可用，模拟空识别")
            text = ""
        t1 = time.time()
        print(f"[延迟] ASR完成 耗时{t1-t0:.2f}s 识别:{text[:30] if text else '(空)'}")
        return jsonify({"text": text or "", "error": None})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"text": "", "error": str(e)}), 500

# ── 单独TTS ──
@app.route('/tts', methods=['POST'])
def tts_route():
    try:
        data = request.json or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "文字不能为空"}), 400
        audio = tts.generate(text)
        return jsonify({"audio": audio, "error": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 对话历史存储 ──
import uuid
from datetime import datetime
import glob

CONV_DIR = os.path.join(os.path.dirname(__file__), 'conversations')
os.makedirs(CONV_DIR, exist_ok=True)

# 获取历史对话列表
@app.route('/conversations', methods=['GET'])
def list_conversations():
    try:
        files = sorted(
            glob.glob(os.path.join(CONV_DIR, '*.json')),
            key=os.path.getmtime, reverse=True
        )
        result = []
        for f in files[:50]:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            result.append({
                'id': data.get('id'),
                'title': data.get('title', '未命名对话'),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify([])

# 获取单条对话详情
@app.route('/conversations/<conv_id>', methods=['GET'])
def get_conversation(conv_id):
    try:
        path = os.path.join(CONV_DIR, f'{conv_id}.json')
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except:
        return jsonify({'error': 'not found'}), 404

# 保存/更新对话
@app.route('/conversations', methods=['POST'])
def save_conversation():
    try:
        data = request.json or {}
        conv_id = data.get('id') or str(uuid.uuid4())[:8]
        messages = data.get('messages', [])

        title = '新对话'
        for msg in messages:
            if msg.get('role') == 'user':
                title = msg.get('content', '')[:15]
                if len(msg.get('content', '')) > 15:
                    title += '...'
                break

        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        path = os.path.join(CONV_DIR, f'{conv_id}.json')

        created_at = now
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                old = json.load(f)
            created_at = old.get('created_at', now)

        payload = {
            'id': conv_id,
            'title': title,
            'created_at': created_at,
            'updated_at': now,
            'messages': messages,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return jsonify({'id': conv_id, 'title': title})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 删除对话
@app.route('/conversations/<conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    try:
        path = os.path.join(CONV_DIR, f'{conv_id}.json')
        if os.path.exists(path):
            os.remove(path)
        return jsonify({'ok': True})
    except:
        return jsonify({'error': 'failed'}), 500

if __name__ == '__main__':
    # 云端部署（Railway）不需要日志重定向和语音预加载
    if os.environ.get('RAILWAY_ENVIRONMENT'):
        port = int(os.environ.get('PORT', 8080))
        print(f"[启动] Railway 模式，端口 {port}", flush=True)
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=4)
    else:
        # 本地开发模式：日志重定向 + 语音预加载
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mks_debug.log')
        sys.stdout = open(log_file, 'w', encoding='utf-8')
        sys.stderr = sys.stdout
        print(f"[启动] 日志重定向到 {log_file}", flush=True)

        # 预加载语音模型（避免首次请求超时）
        import threading
        def _preload():
            print("[启动] 预加载语音识别模型...")
            try:
                import asr
                if _HAS_ASR:
                    m = asr.get_model()
                    if m:
                        print("[启动] 语音模型加载完成")
                    else:
                        print("[启动] 语音模型加载失败")
                else:
                    print("[启动] ASR 不可用，跳过语音预加载")
            except Exception as e:
                print(f"[启动] 预加载异常: {e}")
                import traceback
                traceback.print_exc()
        threading.Thread(target=_preload, daemon=True).start()

        print("=" * 45)
        print(" [MKS] Marx Agent 启动")
        print(" URL: http://localhost:5700")
        print(" 接口: /chat /chat_voice /chat_stream /tts")
        print("=" * 45)
        from waitress import serve
        serve(app, host='0.0.0.0', port=5700, threads=8)
