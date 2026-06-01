# -*- coding: utf-8 -*-
"""
马克思人格智能体 — 后端服务 v3（崩溃保护版）

安装依赖：pip install -r requirements.txt
运行：python app.py

首次使用前需设置环境变量或创建 .env 文件：
  DEEPSEEK_API_KEY=sk-your-key-here
"""

import sys
import os
import json
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import io
import base64
import re
import time
import traceback
import pathlib
import tempfile
import subprocess
import threading

from flask import Flask, request, jsonify, send_from_directory
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

# ── API Key 是否是默认占位值 ──
def _is_default_key(key):
    """检查 API Key 是否还是默认占位值"""
    return not key or key.startswith("sk-") and any(ord(c) > 127 for c in key)

# ── 可选依赖：导入失败不崩溃，只打警告 ──
EDGE_TTS_OK = False
PYTTSX3_OK = False
NUMPY_OK = False
VAD_OK = False
WHISPER_OK = False
SOCK_OK = False

try:
    import edge_tts
    EDGE_TTS_OK = True
except ImportError:
    pass

try:
    import pyttsx3
    PYTTSX3_OK = True
except ImportError:
    pass

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    pass

try:
    import webrtcvad
    VAD_OK = True
except ImportError:
    pass

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    pass

# WebSocket 支持
SOCK_OK = True

app_dir = pathlib.Path(__file__).parent.resolve()
frontend_dir = os.path.join(app_dir, 'frontend')
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
# ============================================================
# 配置区
# ============================================================
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME       = "deepseek-chat"
VOICE_NAME  = "zh-CN-YunjianNeural"
VOICE_RATE  = "+5%"
VOICE_PITCH = "-8Hz"

# ── ffmpeg（懒查找） ──
_FFMPEG_PATH = None
def find_ffmpeg():
    global _FFMPEG_PATH
    if _FFMPEG_PATH:
        return _FFMPEG_PATH
    try:
        r = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = r.stdout.strip().split('\n')[0].strip()
            if p and os.path.isfile(p):
                _FFMPEG_PATH = p
                return _FFMPEG_PATH
    except: pass
    try:
        r = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if r.returncode == 0:
            _FFMPEG_PATH = 'ffmpeg'
            return _FFMPEG_PATH
    except: pass
    return _FFMPEG_PATH

# ── ASR 模型（完全懒加载，启动时不碰 torch） ──
WHISPER_MODEL = None
WHISPER_DEVICE = None

def get_whisper_model():
    global WHISPER_MODEL, WHISPER_DEVICE
    if WHISPER_MODEL is not None:
        return WHISPER_MODEL
    if not WHISPER_OK:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            WHISPER_DEVICE = "cuda"
            device = "cuda"
            compute_type = "float16"
            torch.cuda.set_per_process_memory_fraction(0.55)
            print(f"[ASR] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory//1024**3}GB)")
        else:
            WHISPER_DEVICE = "cpu"
            device = "cpu"
            compute_type = "int8"
            print("[ASR] GPU不可用，使用CPU")
        print(f"[ASR] 加载 faster-whisper small, device={device}, compute={compute_type}")
        WHISPER_MODEL = WhisperModel("small", device=device, compute_type=compute_type)
        print(f"[ASR] 加载完成")
        return WHISPER_MODEL
    except Exception as e:
        print(f"[ASR] 加载失败: {e}")
        traceback.print_exc()
        return None

def safe_transcribe(model, samples_np):
    try:
        if model is None or samples_np is None or len(samples_np) == 0:
            return ""
        segments, _ = model.transcribe(
            samples_np, language="zh", beam_size=1,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300)
        )
        return "".join(seg.text for seg in segments).strip()
    except Exception as e:
        print(f"[ASR] 转写异常: {e}")
        return ""

def decode_audio_to_float32(audio_bytes):
    try:
        if len(audio_bytes) < 100:
            return None
        ff = r'C:\Users\武旭龙\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe'
        if not os.path.isfile(ff):
            print("[audio] ffmpeg未安装")
            return None
        suffix = '.wav' if audio_bytes[:4] == b'RIFF' else '.webm'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            in_path = f.name
        out_path = in_path + '_out.wav'
        try:
            r = subprocess.run([ff, '-y', '-i', in_path, '-ar', '16000', '-ac', '1', '-f', 'wav', out_path],
                               capture_output=True, timeout=15)
            if r.returncode != 0:
                print(f"[audio] ffmpeg失败: {r.stderr.decode(errors='ignore')[:100]}")
                return None
            with open(out_path, 'rb') as fw:
                import wave as wavlib
                with wavlib.open(io.BytesIO(fw.read()), 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(frames, dtype=np.int16)
                    return samples.astype(np.float32) / 32768.0
        except subprocess.TimeoutExpired:
            print("[audio] ffmpeg超时")
            return None
        except Exception as e:
            print(f"[audio] 解码异常: {e}")
            return None
        finally:
            for p in [in_path, out_path]:
                try: os.unlink(p)
                except: pass
    except Exception as e:
        print(f"[audio] 顶层异常: {e}")
        return None

def transcribe_audio(pcm_data, sample_rate):
    try:
        model = get_whisper_model()
        if model is None:
            return ""
        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        return safe_transcribe(model, audio_array)
    except Exception as e:
        print(f"[ASR] transcribe_audio失败: {e}")
        return ""

# ── DeepSeek API ──
MARX_SYSTEM_PROMPT = """你是卡尔·马克思（Karl Marx，1818-1883），哲学家、经济学家、革命理论家。此刻在伦敦书房与来访者对话。

【核心】
- 从物质条件与生产关系出发，坚持辩证法和历史唯物主义
- 语言严谨有力，偶带嘲讽，喜欢反问
- 对工人阶级有深切同情，对资本逻辑有冷静批判
- 回答必须简短有力，不超过100字
- 用辩证法一针见血，直指问题本质
- 带有德意志式的理性与克制的幽默
- 像在私下谈话，不是在做演讲
- 禁止长篇大论，禁止列举条目
- 结尾可以反问，引发思考
- 如果用户消息以【思辨:】开头：用反问开头，不直接给答案
- 如果用户消息以【引用:】开头：结合著作精神阐发
- 如果用户没有使用任何前缀：正常回答，不要加【思辨:】或【引用:】开头
- 不说"作为AI"、"作为语言模型"
- 不说"好的"、"当然"、"很高兴为您解答！"
- 不确定引文时说"我的大意是..."而非假装精确引用
- 禁止Markdown标记，禁止括号动作描述

【语音对话模式】
当用户以语音方式提问时，回答必须：
- 不超过2句话，每句不超过25字
- 用口语化表达，像真实对话
- 不用书面语、不用列举条目
- 可以用反问结尾引发思考
- 绝对不能有任何标点以外的特殊符号"""

def call_deepseek(user_message, history=None):
    messages = [{"role": "system", "content": MARX_SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-(6*2):])
    messages.append({"role": "user", "content": user_message})
    headers = {"Authorization": "Bearer " + DEEPSEEK_API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME, "messages": messages,
        "temperature": 0.75, "max_tokens": 400, "top_p": 0.9,
        "presence_penalty": 0.3, "stream": False
    }
    try:
        import requests
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[DeepSeek] 调用失败: {e}")
        return None

# ── TTS 缓存 ──
_tts_cache = {}
_tts_cache_max = 50

def _tts_cache_get(text):
    return _tts_cache.get(text)

def _tts_cache_set(text, audio_b64):
    if len(_tts_cache) >= _tts_cache_max:
        key = next(iter(_tts_cache))
        del _tts_cache[key]
    _tts_cache[text] = audio_b64

def clean_text_for_tts(text):
    clean = text
    clean = re.sub(r'\*+(.+?)\*+', r'\1', clean)
    clean = re.sub(r'【.+?:】', '', clean)
    clean = re.sub(r'\n{3,}', '\n\n', clean)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = re.sub(r'\s{2,}', ' ', clean).strip()
    return clean

async def generate_tts(text):
    cached = _tts_cache_get(text)
    if cached:
        return cached
    clean = clean_text_for_tts(text)
    if EDGE_TTS_OK:
        try:
            communicate = edge_tts.Communicate(text=clean, voice=VOICE_NAME, rate=VOICE_RATE, pitch=VOICE_PITCH, proxy=None)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            if buf.tell() > 0:
                buf.seek(0)
                audio_b64 = base64.b64encode(buf.read()).decode()
                _tts_cache_set(text, audio_b64)
                return audio_b64
        except Exception as e:
            print(f"[TTS] EdgeTTS失败: {e}")
    if PYTTSX3_OK:
        try:
            engine = pyttsx3.init()
            for v in engine.getProperty('voices'):
                if 'zh' in v.id.lower() or 'chinese' in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
            engine.setProperty('rate', 130)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp_path = f.name
            engine.save_to_file(clean, tmp_path)
            engine.runAndWait()
            with open(tmp_path, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            os.unlink(tmp_path)
            _tts_cache_set(text, audio_b64)
            return audio_b64
        except Exception as e:
            print(f"[TTS] pyttsx3也失败: {e}")
    return None

# ============================================================
# 路由
# ============================================================
@app.after_request
def add_cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    return resp

@app.route('/')
def index():
    try:
        return send_from_directory(frontend_dir, 'index.html')
    except Exception as e:
        print(f"[静态文件] 访问失败: {e}")
        return "服务启动中，请稍后刷新", 503

@app.route('/health')
def health():
    return jsonify({"status": "running", "model": MODEL_NAME, "voice": VOICE_NAME})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not found"}), 404

@app.errorhandler(500)
def server_error(e):
    traceback.print_exc()
    return jsonify({"error": "server error"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "请求体为空", "reply": None}), 400
        user_message = data.get('message', '').strip()
        history = data.get('history', [])
        need_tts = data.get('tts', False)
        if not user_message:
            return jsonify({"error": "消息为空", "reply": None}), 400
        messages = [{"role": "system", "content": MARX_SYSTEM_PROMPT}]
        messages.extend(history[-(10*2):])
        messages.append({"role": "user", "content": user_message})
        headers = {"Authorization": "Bearer " + DEEPSEEK_API_KEY, "Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME, "messages": messages,
            "temperature": 0.75, "max_tokens": 400, "top_p": 0.9,
            "presence_penalty": 0.3, "stream": True
        }
        import requests
        reply = ""
        with requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                chunk = line[6:]
                if chunk == b"[DONE]":
                    break
                try:
                    reply += json.loads(chunk)["choices"][0]["delta"].get("content", "")
                except:
                    pass
        audio_b64 = None
        if need_tts and reply:
            try:
                audio_b64 = asyncio.run(generate_tts(reply))
            except Exception as e:
                print(f"[Chat] TTS失败: {e}")
        return jsonify({"reply": reply, "audio": audio_b64, "error": None})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "reply": None}), 500

@app.route('/chat_voice', methods=['POST'])
def chat_voice():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "请求体为空"}), 400
        audio_b64 = data.get('audio_b64', '')
        history = data.get('history', [])
        need_tts = data.get('tts', True)
        if not audio_b64:
            return jsonify({"error": "音频为空"}), 400
        audio_bytes = base64.b64decode(audio_b64)
        print(f"[chat_voice] 收到音频 {len(audio_bytes)} bytes")
        samples_np = decode_audio_to_float32(audio_bytes)
        if samples_np is None or len(samples_np) < 1600:
            return jsonify({"reply": "音频解码失败", "audio": None, "error": None})
        model = get_whisper_model()
        if not model:
            return jsonify({"reply": "语音识别模型未加载", "audio": None, "error": None})
        text = safe_transcribe(model, samples_np)
        print(f"[chat_voice] 识别: '{text}'")
        if not text:
            return jsonify({"reply": "未识别到语音内容", "audio": None, "error": None})
        reply = call_deepseek(text, history)
        if not reply:
            return jsonify({"error": "AI回复失败", "reply": None, "audio": None}), 500
        print(f"[chat_voice] DeepSeek: {reply[:50]}")
        audio_result = None
        if need_tts:
            try:
                audio_result = asyncio.run(generate_tts(reply))
            except Exception as e:
                print(f"[chat_voice] TTS失败: {e}")
        return jsonify({"reply": reply, "audio": audio_result, "error": None})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "reply": None, "audio": None}), 500

@app.route('/tts', methods=['POST'])
def tts():
    try:
        data = request.json
        text = data.get('text', '').strip() if data else ''
        if not text:
            return jsonify({"error": "文字不能为空"}), 400
        audio_b64 = asyncio.run(generate_tts(text))
        return jsonify({"audio": audio_b64, "error": None})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/test_tts')
def test_tts():
    try:
        audio = asyncio.run(generate_tts("哲学的任务不是解释世界，而是改变世界。"))
        return jsonify({"success": True, "audio_length": len(audio)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ── WebSocket（gevent 原生支持，不依赖 flask_sock） ──
if SOCK_OK:
    import collections
    import struct as _struct

    def ws_app(environ, start_response):
        """WSGI应用：拦截 /ws/voice 的 WebSocket 升级请求"""
        if environ.get('PATH_INFO') == '/ws/voice':
            ws = environ['wsgi.websocket']
            handle_websocket(ws)
            return []
        return app(environ, start_response)

    def handle_websocket(ws):
        """处理 /ws/voice 的 WebSocket 连接"""
        try:
            ws.send(json.dumps({"type": "status", "text": "connected"}))
            audio_buffer = collections.deque()
            last_activity = time.time()
            user_idle = False
            while True:
                try:
                    data = ws.receive()
                    if data is None:
                        break
                    if isinstance(data, bytes) and len(data) > 0:
                        audio_buffer.append(data)
                        last_activity = time.time()
                        samples = _struct.unpack_from('<' + 'h' * (len(data)//2), data)
                        rms = (sum(s*s for s in samples)/len(samples))**0.5
                        if rms > 500:
                            user_idle = False
                        else:
                            if not user_idle and len(audio_buffer) > 0:
                                elapsed = time.time() - last_activity
                                if elapsed > 0.6 and rms < 300:
                                    _stream_ws_reply(ws, audio_buffer)
                                    audio_buffer.clear()
                                    user_idle = True
                    elif isinstance(data, str):
                        try:
                            msg = json.loads(data)
                            if msg.get("type") == "ping":
                                ws.send(json.dumps({"type": "pong"}))
                        except:
                            pass
                except Exception:
                    break
        except Exception as e:
            print(f"[ws] 异常: {e}")
        finally:
            try: ws.close()
            except: pass

    # 注册 WS 路由到 Flask（gevent 接管）
    _WS_QUEUE = 0

    def _stream_ws_reply(ws, buf):
        global _WS_QUEUE
        if _WS_QUEUE:
            return
        _WS_QUEUE += 1
        try:
            raw = b''.join(buf)
            if len(raw) < 3200:
                return
            model = get_whisper_model()
            if not model:
                return
            samples_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            text = safe_transcribe(model, samples_np)
            if not text:
                return
            ws.send(json.dumps({"type": "transcript", "text": text}))
            asyncio.run(_ws_stream_deepseek(ws, text))
        except Exception as e:
            print(f"[ws] 转写错误: {e}")
        finally:
            _WS_QUEUE -= 1

    async def _ws_stream_deepseek(ws, text):
        try:
            messages = [{"role": "system", "content": MARX_SYSTEM_PROMPT},
                        {"role": "user", "content": text}]
            headers = {"Authorization": "Bearer " + DEEPSEEK_API_KEY, "Content-Type": "application/json"}
            payload = {
                "model": MODEL_NAME, "messages": messages,
                "temperature": 0.75, "max_tokens": 400, "top_p": 0.9,
                "presence_penalty": 0.3, "stream": True
            }
            import requests
            reply_so_far = ""
            sentence_buf = ""
            with requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data: "):
                        continue
                    chunk = line[6:]
                    if chunk == b"[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                    except:
                        continue
                    if not delta:
                        continue
                    sentence_buf += delta
                    reply_so_far += delta
                    if any(c in delta for c in "。！？…\n"):
                        sentence = sentence_buf.strip()
                        sentence_buf = ""
                        if sentence:
                            ws.send(json.dumps({"type": "transcript_reply", "text": sentence}))
                            audio = await generate_tts(sentence)
                            if audio:
                                ws.send(json.dumps({"type": "audio_chunk", "audio": audio, "text": sentence}))
            if sentence_buf.strip():
                sentence = sentence_buf.strip()
                ws.send(json.dumps({"type": "transcript_reply", "text": sentence}))
                audio = await generate_tts(sentence)
                if audio:
                    ws.send(json.dumps({"type": "audio_chunk", "audio": audio, "text": sentence}))
            ws.send(json.dumps({"type": "done", "full_text": reply_so_far}))
        except Exception as e:
            print(f"[ws] 流式回复失败: {e}")

# ── 启动（极度精简，不加载任何资源） ──
if __name__ == '__main__':
    print("=" * 40)
    print(" [MKS] Marx Agent v3")
    print(" URL: http://localhost:5700")
    if DEEPSEEK_API_KEY:
        print(" [OK] DeepSeek API Key ready")
    else:
        print(" [WARN] DeepSeek API Key not configured!")
    print(" [..] 所有资源第一次使用时才加载")
    print("=" * 40)
    server = pywsgi.WSGIServer(('127.0.0.1', 5700), ws_app, handler_class=WebSocketHandler)
    print(" [OK] WebSocket 就绪 (gevent)")
    server.serve_forever()


