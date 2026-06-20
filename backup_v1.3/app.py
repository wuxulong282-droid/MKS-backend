from flask import Flask, request, jsonify, send_from_directory, Response
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
import asr
import tts
import llm

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# ── 静态文件 ──
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not found"}), 404

# ── 健康检查 ──
@app.route('/health')
def health_check():
    return jsonify({"status":"ok"})

@app.route('/test_500')
def test_500():
    """Force a 500 with print"""
    print("[test] about to crash", flush=True)
    sys.stderr.write("[test] stderr before crash\n")
    sys.stderr.flush()
    raise ValueError("test crash")
    return jsonify({"ok":True})
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
            sentence_buf = ""
            full_reply = ""
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
                        audio = tts.generate(sentence)
                        if audio:
                            yield f"data: {json.dumps({'type': 'audio', 'audio': audio, 'text': sentence})}\n\n"

            if sentence_buf.strip():
                audio = tts.generate(sentence_buf.strip())
                if audio:
                    yield f"data: {json.dumps({'type': 'audio', 'audio': audio, 'text': sentence_buf.strip()})}\n\n"

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
    try:
        data = request.json or {}
        audio_b64 = data.get('audio_b64', '') or data.get('audio', '')
        history = data.get('history', [])

        if not audio_b64:
            return jsonify({"reply": "没有收到音频", "audio": None}), 200

        audio_bytes = base64.b64decode(audio_b64)
        print(f"[voice] 收到音频 {len(audio_bytes)} bytes")

        text = asr.transcribe(audio_bytes)
        if not text:
            return jsonify({"reply": "", "audio": None}), 200

        print(f"[voice] 识别: {text}")

        reply = ""
        for chunk in llm.chat_stream(
            text, history, MARX_SYSTEM_PROMPT,
            DEEPSEEK_API_KEY, DEEPSEEK_API_URL, MODEL_NAME
        ):
            reply += chunk

        audio = tts.generate(reply)

        return jsonify({
            "transcript": text,
            "reply": reply,
            "audio": audio,
            "error": None
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        _crash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voice_crash.txt')
        with open(_crash_path, 'w', encoding='utf-8') as _f:
            _f.write(tb)
        traceback.print_exc()
        return jsonify({"reply": "处理出错，请重试", "audio": None, "error": None}), 200

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

if __name__ == '__main__':
    # 重定向日志到文件（因为 Waitress 可能吞掉 print）
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
            m = asr.get_model()
            if m:
                print("[启动] 语音模型加载完成")
            else:
                print("[启动] 语音模型加载失败")
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
