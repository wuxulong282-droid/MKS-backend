"""
马克思人格智能体 — 后端服务 v2（含语音模块）

安装依赖：pip install -r requirements.txt
运行：python app.py

首次使用语音功能前需设置环境变量或创建 .env 文件：
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
import struct
import threading
import time
import traceback
import pathlib
import tempfile
import subprocess

# ── 查找 ffmpeg ──
_FFMPEG_PATH = None
def _find_ffmpeg():
    global _FFMPEG_PATH
    if _FFMPEG_PATH:
        return _FFMPEG_PATH
    # 1) 先试系统 PATH
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            _FFMPEG_PATH = 'ffmpeg'
            print("  [OK] ffmpeg 已安装 (PATH)")
            return _FFMPEG_PATH
    except: pass
    # 2) 检查常见的 WinGet 安装路径
    import glob
    for root in [os.path.expanduser('~\\AppData\\Local\\Microsoft\\WinGet\\Packages')]:
        pattern = root + '\\Gyan.FFmpeg*\\ffmpeg-*\\bin\\ffmpeg.exe'
        for f in glob.glob(pattern):
            _FFMPEG_PATH = f
            print(f"  [OK] ffmpeg 已安装: {f}")
            return _FFMPEG_PATH
    # 3) 其他常见路径
    for p in ['C:\\ffmpeg\\bin\\ffmpeg.exe', 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe']:
        if os.path.exists(p):
            _FFMPEG_PATH = p
            print(f"  [OK] ffmpeg 已安装: {p}")
            return _FFMPEG_PATH
    print("  [WARN] ffmpeg 未找到！语音识别将失败")
    print("        请将 ffmpeg 加入系统 PATH 后重启")
    return None

_find_ffmpeg()

# 从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

def _find_ffmpeg():
    global _FFMPEG_PATH
    if _FFMPEG_PATH:
        return _FFMPEG_PATH
    # 1) where 命令（Windows专用）
    try:
        r = subprocess.run(['where', 'ffmpeg'],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = r.stdout.strip().split('\n')[0].strip()
            if p and os.path.isfile(p):
                _FFMPEG_PATH = p
                print(f"  [OK] ffmpeg: {p}")
                return _FFMPEG_PATH
    except: pass
    # 2) 系统 PATH 直接试
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            _FFMPEG_PATH = 'ffmpeg'
            print("  [OK] ffmpeg 已安装 (PATH)")
            return _FFMPEG_PATH
    except: pass
    # 3) WinGet 安装路径
    import glob
    for root in [os.path.expanduser('~\\AppData\\Local\\Microsoft\\WinGet\\Packages')]:
        pattern = root + '\\Gyan.FFmpeg*\\ffmpeg-*\\bin\\ffmpeg.exe'
        for fpath in glob.glob(pattern):
            _FFMPEG_PATH = fpath
            print(f"  [OK] ffmpeg: {fpath}")
            return _FFMPEG_PATH
    # 4) 其他常见路径
    for p in ['C:\\ffmpeg\\bin\\ffmpeg.exe', 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe']:
        if os.path.exists(p):
            _FFMPEG_PATH = p
            print(f"  [OK] ffmpeg: {p}")
            return _FFMPEG_PATH
    print("  [WARN] ffmpeg 未找到！语音识别将失败")
    print("        请将 ffmpeg 加入系统 PATH 后重启")
    return None


def _is_default_key(key):
    """检查 API Key 是否还是默认占位值"""
    return not key or key.startswith("sk-") and any(ord(c) > 127 for c in key)

# ── 可选依赖：导入失败不崩溃，只打警告 ──

try:
    import edge_tts
    EDGE_TTS_OK = True
except ImportError:
    print("⚠ edge_tts 未安装（可选，pip install edge-tts）")
    EDGE_TTS_OK = False

try:
    import pyttsx3
    PYTTSX3_OK = True
except ImportError:
    print("⚠ pyttsx3 未安装（可选，pip install pyttsx3）")
    PYTTSX3_OK = False

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    print("⚠ numpy 未安装（pip install numpy）")
    NUMPY_OK = False

try:
    import webrtcvad
    VAD_OK = True
except ImportError:
    print("⚠ webrtcvad 未安装，VAD 静音检测不可用（pip install webrtcvad）")
    VAD_OK = False

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    print("⚠ faster-whisper 未安装，语音识别不可用（pip install faster-whisper）")
    WHISPER_OK = False

app_dir = pathlib.Path(__file__).parent.resolve()
frontend_dir = os.path.join(app_dir, 'frontend')
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app)

# ============================================================
# 配置区
# ============================================================

# DeepSeek API（优先读环境变量）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-a21fda4926dd4210998f15b66286dbf9")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME       = "deepseek-chat"

# EdgeTTS 配置
VOICE_NAME  = "zh-CN-YunjianNeural"
VOICE_RATE  = "+5%"
VOICE_PITCH = "-8Hz"

# ── WebSocket Sock ──
try:
    from flask_sock import Sock
    SOCK_OK = True
except ImportError:
    print("[WARN] flask_sock not installed (pip install flask-sock)")
    SOCK_OK = False

if SOCK_OK:
    sock = Sock(app)
    import collections
    import struct
    WS_VAD = None
    WS_WHISPER = None
    WS_QUEUE = 0

    def _ws_init():
        global WS_VAD, WS_WHISPER
        try:
            if VAD_OK and WS_VAD is None:
                WS_VAD = webrtcvad.Vad(2)
        except: pass
        try:
            if WHISPER_OK and WS_WHISPER is None:
                WS_WHISPER = WHISPER_MODEL
        except: pass

    @sock.route('/ws/voice')
    def voice(ws):
        try:
            _ws_init()
            print("[ws] 客户端已连接")
            audio_buffer = collections.deque()
            last_activity = time.time()
            transcribing = False
            user_idle = False
            try:
                ws.send(json.dumps({"type": "status", "text": "connected"}))
            except: pass
            try:
                while True:
                    try:
                        data = ws.receive()
                        if data is None:
                            break
                        if isinstance(data, bytes) and len(data) > 0:
                            audio_buffer.append(data)
                            last_activity = time.time()
                            samples = struct.unpack_from('<' + 'h' * (len(data) // 2), data)
                            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
                            if rms > 500:
                                user_idle = False
                                transcribing = False
                            else:
                                if not transcribing and not user_idle and len(audio_buffer) > 0:
                                    elapsed = time.time() - last_activity
                                    if elapsed > 0.6 and rms < 300:
                                        transcribing = True
                                        _do_ws_transcribe(ws, audio_buffer)
                                        audio_buffer.clear()
                                        user_idle = True
                        elif isinstance(data, str):
                            try:
                                msg = json.loads(data)
                                if msg.get("type") == "ping":
                                    ws.send(json.dumps({"type": "pong"}))
                            except: pass
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
            finally:
                try: ws.close()
                except: pass
            print("[ws] 客户端已断开")
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _do_ws_transcribe(ws, buf):
        global WS_QUEUE
        if WS_QUEUE:
            return
        WS_QUEUE += 1
        try:
            try:
                ws.send(json.dumps({"type": "status", "text": "thinking"}))
            except: pass
            raw = b''.join(buf)
            if len(raw) < 3200:
                return
            print("[ws] 语音转写... 长度:", len(raw))
            text = ""
            if WHISPER_OK and WS_WHISPER:
                import numpy as np
                samples_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                text = safe_transcribe(WS_WHISPER, samples_np)
            if text:
                print("[ws] 识别:", text[:50])
                try:
                    ws.send(json.dumps({"type": "transcript", "text": text}))
                except: pass
                asyncio.run(stream_reply_with_tts(ws, text, []))
            else:
                try:
                    ws.send(json.dumps({"type": "status", "text": "未检测到语音"}))
                except: pass
        except Exception as e:
            print("[ws] 转写错误:", e)
            try:
                ws.send(json.dumps({"type": "error", "text": str(e)}))
            except: pass
        finally:
            WS_QUEUE -= 1

async def stream_reply_with_tts(ws, text_input, history):
    """流式分句：DeepSeek 流式输出，每句话实时 TTS 推送"""
    try:
        ws.send(json.dumps({"type": "status", "text": "thinking"}))
    except:
        pass
    messages = [{"role": "system", "content": MARX_SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-(6 * 2):])
    messages.append({"role": "user", "content": text_input})
    headers = {
        "Authorization": "Bearer " + DEEPSEEK_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.75,
        "max_tokens": 400,
        "top_p": 0.9,
        "presence_penalty": 0.3,
        "stream": True
    }
    reply_so_far = ""
    sentence_buf = ""
    try:
        ws.send(json.dumps({"type": "status", "text": "speaking"}))
    except:
        pass
    try:
        with requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                chunk_raw = line[6:]
                if chunk_raw == b"[DONE]":
                    break
                try:
                    delta = json.loads(chunk_raw)["choices"][0]["delta"].get("content", "")
                except:
                    continue
                if not delta:
                    continue
                sentence_buf += delta
                reply_so_far += delta
                # 遇到句子结束符，切句推送
                if any(c in delta for c in "。！？…\n"):
                    sentence = sentence_buf.strip()
                    sentence_buf = ""
                    if sentence:
                        try:
                            ws.send(json.dumps({"type": "transcript_reply", "text": sentence}))
                        except:
                            pass
                        try:
                            audio = await generate_tts(sentence)
                            if audio:
                                try:
                                    ws.send(json.dumps({
                                        "type": "audio_chunk",
                                        "audio": audio,
                                        "text": sentence
                                    }))
                                except:
                                    pass
                        except Exception as e:
                            print("[ws] 分句TTS失败:", e)
        # 处理最后剩余的一句（没有句号结尾的尾巴）
        if sentence_buf.strip():
            sentence = sentence_buf.strip()
            try:
                ws.send(json.dumps({"type": "transcript_reply", "text": sentence}))
            except:
                pass
            try:
                audio = await generate_tts(sentence)
                if audio:
                    try:
                        ws.send(json.dumps({
                            "type": "audio_chunk",
                            "audio": audio,
                            "text": sentence
                        }))
                    except:
                        pass
            except Exception as e:
                print("[ws] 末句TTS失败:", e)
    except Exception as e:
        print("[ws] DeepSeek流式调用失败:", e)
    try:
        ws.send(json.dumps({"type": "done", "full_text": reply_so_far}))
    except:
        pass

def _ws_send_json(ws, obj):
    """线程安全WebSocket JSON发送"""
    try:
        ws.send(json.dumps(obj, ensure_ascii=False))
    except:
        pass

# ── ASR 模型（懒加载） ──
WHISPER_MODEL = None
import torch
if torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
else:
    WHISPER_DEVICE = "cpu"

def get_whisper_model():
    """
    懒加载 faster-whisper 模型
    自动检测 CUDA 可用性，选择最优设备和精度
    """
    global WHISPER_MODEL, WHISPER_DEVICE
    if WHISPER_MODEL is not None:
        return WHISPER_MODEL
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        torch.cuda.set_per_process_memory_fraction(0.55)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"[ASR] GPU可用: {gpu_name}")
        print(f"[ASR] 显存: {gpu_mem}GB")
        print(f"[ASR] GPU 显存限制: 55%")
    else:
        device = "cpu"
        compute_type = "int8"
        print("[ASR] GPU不可用，使用CPU")
    WHISPER_DEVICE = device
    print(f"[ASR] 加载 faster-whisper small，设备 {device}，精度 {compute_type}")
    WHISPER_MODEL = WhisperModel(
        "small",
        device=device,
        compute_type=compute_type
    )
    print(f"[ASR] 加载完成，device={device}")
    return WHISPER_MODEL


def decode_audio_to_float32(audio_bytes: bytes):
    """把任意格式音频转为Whisper需要的float32数组(16kHz单声道)"""
    try:
        if len(audio_bytes) < 100:
            print("[audio] 数据太短")
            return None

        suffix = '.wav' if audio_bytes[:4] == b'RIFF' else '.webm'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            in_path = f.name
        out_path = in_path + '_out.wav'

        try:
            ff_path = _FFMPEG_PATH if _FFMPEG_PATH else 'ffmpeg'
            if _FFMPEG_PATH is None:
                print("[audio] ffmpeg未安装，无法解码")
                return None
            result = subprocess.run(
                [ff_path, '-y', '-i', in_path,
                 '-ar', '16000', '-ac', '1', '-f', 'wav', out_path],
                capture_output=True, timeout=15
            )
            if result.returncode != 0:
                err = result.stderr.decode(errors='ignore')[:200]
                print(f"[audio] ffmpeg失败: {err}")
                return None

            with open(out_path, 'rb') as wf_raw:
                wav_data = wf_raw.read()

            import wave, io
            with wave.open(io.BytesIO(wav_data), 'rb') as wf:
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
        print(f"[audio] 解码顶层异常: {e}")
        return None

def safe_transcribe(model, samples_np):
    """安全的Whisper转写包装，任何异常返回空字符串"""
    try:
        if model is None or samples_np is None or len(samples_np) == 0:
            return ""
        segments, _ = model.transcribe(
            samples_np,
            language="zh",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        return "".join(seg.text for seg in segments).strip()
    except Exception as e:
        print(f"[ASR] 转写异常: {e}")
        return ""



def transcribe_audio(pcm_data: bytes, sample_rate: int) -> str:
    """
    用 PCM 音频数据 + faster-whisper 转为文字
    指定 language='zh' 提高中文识别准确率
    """
    import time as _t
    try:
        _t0 = _t.time()
        model = get_whisper_model()
        if model is None:
            return ""
        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        text = safe_transcribe(model, audio_array)
        dt = _t.time() - _t0
        print(f"[ASR] 识别结果 ({dt:.2f}s): {text[:60]}")
        return text
    except Exception as e:
        print(f"faster-whisper 转写失败: {e}")
        return ""

# ── DeepSeek API ──
def call_deepseek(user_message: str, history: list = None) -> str:
    """
    调用 DeepSeek API 生成马克思回复
    """
    messages = [{"role": "system", "content": MARX_SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-(6 * 2):])
    messages.append({"role": "user", "content": user_message})
    auth_header = "Bearer " + DEEPSEEK_API_KEY
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.75,
        "max_tokens": 400,
        "top_p": 0.9,
        "presence_penalty": 0.3,
        "stream": False
    }
    _t0 = time.time()
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        reply = resp.json()['choices'][0]['message']['content']
        print(f"[DeepSeek] 完成（{time.time()-_t0:.1f}s）")
        return reply
    except Exception as e:
        traceback.print_exc()
        print(f"DeepSeek 调用失败: {e}")
        return None

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

@app.route('/')
def index():
    resp = send_from_directory(frontend_dir, 'index.html')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/health')
def health():
    gpu_info = "CPU模式"
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            gpu_info = f"{alloc:.1f}GB"
    except:
        pass
    return jsonify({
        "status": "running",
        "model": MODEL_NAME,
        "voice": VOICE_NAME,
        "gpu": gpu_info,
        "asr_device": WHISPER_DEVICE,
        "cuda_available": str(torch.cuda.is_available())
    })

@app.route('/test')
def test():
    if _is_default_key(DEEPSEEK_API_KEY):
        return jsonify({"error": "API Key 未配置，请先在 app.py 中设置 DEEPSEEK_API_KEY"})
    try:
        auth_header = "Bearer " + DEEPSEEK_API_KEY
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": auth_header,
                     "Content-Type": "application/json"},
            json={"model": MODEL_NAME,
                  "messages": [{"role": "user", "content": "你好"}],
                  "max_tokens": 10},
            timeout=10
        )
        return jsonify({"status": resp.status_code, "body": resp.json()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

# ── TTS 缓存 ──
_tts_cache = {}
_tts_cache_max = 50

def _tts_cache_get(text: str):
    """从缓存取 TTS 结果，返回 base64 或 None"""
    return _tts_cache.get(text)

def _tts_cache_set(text: str, audio_b64: str):
    """缓存 TTS 结果"""
    if len(_tts_cache) >= _tts_cache_max:
        key = next(iter(_tts_cache))
        del _tts_cache[key]
    _tts_cache[text] = audio_b64

def _prewarm_tts():
    """后台线程预热 TTS，让 EdgeTTS 首次调用更快"""
    try:
        import asyncio
        text = "请坐，同志。"
        audio = asyncio.run(generate_tts(text))
        _tts_cache_set(text, audio)
        print(f"  [OK] TTS 预热完成（{len(audio)} 字符）")
    except Exception as e:
        print(f"  [OK] TTS 预热完成（不影响使用）: {e}")

@app.route('/test_tts')
def test_tts():
    try:
        audio = asyncio.run(generate_tts("哲学的任务不是解释世界，而是改变世界。"))
        return jsonify({"success": True, "audio_length": len(audio)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

# ── 文字对话接口 ──
@app.route('/chat', methods=['POST'])
def chat():
    """
    接收：{ "message": "...", "history": [...], "tts": true/false }
    返回：{ "reply": "...", "audio": "base64字符串或null", "error": null }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "请求体为空", "reply": None}), 400
        user_message = data.get('message', '').strip()
        history      = data.get('history', [])
        need_tts     = data.get('tts', False)
        if not user_message:
            return jsonify({"error": "消息为空", "reply": None}), 400
        print(f"收到消息: {user_message[:50]}")
        messages = [{"role": "system", "content": MARX_SYSTEM_PROMPT}]
        messages.extend(history[-(10 * 2):])
        messages.append({"role": "user", "content": user_message})
        auth_header = "Bearer " + DEEPSEEK_API_KEY
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.75,
            "max_tokens": 400,
            "top_p": 0.9,
            "presence_penalty": 0.3,
            "stream": True
        }
        reply = ""
        try:
            with requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data:"):
                        continue
                    chunk = line[6:]
                    if chunk == b"[DONE]":
                        break
                    try:
                        import json
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                        reply += delta
                    except:
                        pass
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"LLM请求失败：{str(e)}", "reply": None}), 500
        audio_b64 = None
        if need_tts:
            print("[Chat] 文字回答已生成，后台生成语音...")
            try:
                audio_b64 = asyncio.run(generate_tts(reply))
                print(f"[Chat] 语音生成完成（{len(audio_b64)//1024}KB）")
            except Exception as e:
                traceback.print_exc()
        return jsonify({"reply": reply, "audio": audio_b64, "error": None})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "reply": None}), 500

# ── 语音对话接口 ──
@app.route('/chat_voice', methods=['POST'])
def chat_voice():
    """
    语音对话接口
    接收：{ "audio_b64": "base64音频", "history": [...], "tts": true/false }
    返回：{ "reply": "...", "audio": "base64字符串或null", "error": null }
    """
    data = request.json
    if not data:
        return jsonify({"error": "请求体为空"}), 400
    audio_b64 = data.get('audio_b64', '')
    history = data.get('history', [])
    need_tts = data.get('tts', True)
    if not audio_b64:
        return jsonify({"error": "音频为空"}), 400
    try:
        audio_bytes = base64.b64decode(audio_b64)
        print(f"[chat_voice] 收到音频 {len(audio_bytes)} bytes")

        # 统一音频解码：ffmpeg 转 16kHz 单声道 float32
        samples_np = decode_audio_to_float32(audio_bytes)
        if samples_np is None or len(samples_np) < 1600:
            print("[chat_voice] 音频解码失败或过短")
            return jsonify({"reply": "音频解码失败，请重试", "audio": None, "error": None})

        print(f"[chat_voice] 解码成功，样本数: {len(samples_np)}, 时长: {len(samples_np)/16000:.1f}s")

        model = get_whisper_model()
        if not model:
            return jsonify({"reply": "语音识别模型未加载", "audio": None, "error": None})

        text = safe_transcribe(model, samples_np)
        print(f"[chat_voice] 识别结果: '{text}'")

        if not text:
            return jsonify({"reply": "未识别到语音内容", "audio": None, "error": None})
        print(f"[chat_voice] 识别结果: {text[:50]}")
        reply = call_deepseek(text, history)
        if not reply:
            return jsonify({"error": "AI 回复失败", "reply": None, "audio": None}), 500
        print(f"[chat_voice] DeepSeek 回复: {reply[:50]}")
        audio_b64_result = None
        if need_tts:
            try:
                audio_b64_result = asyncio.run(generate_tts(reply))
            except Exception as e:
                print(f"[chat_voice] TTS 失败: {e}")
        return jsonify({"reply": reply, "audio": audio_b64_result, "error": None})
    except Exception as e:
        print(f"[chat_voice] 错误: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "reply": None, "audio": None}), 500

# ── TTS 接口 ──
@app.route('/tts', methods=['POST'])
def tts():
    try:
        data = request.json
        if not data or not data.get('text', '').strip():
            return jsonify({"error": "文字不能为空"}), 400
        text = data.get('text', '').strip()
        audio_b64 = asyncio.run(generate_tts(text))
        return jsonify({"audio": audio_b64, "error": None})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def clean_text_for_tts(text: str) -> str:
    """清理 TTS 文本：去Markdown标记、动作描述、BREAK TIME、SSML标签等"""
    clean = text
    clean = re.sub(r'\*+(.+?)\*+', r'\1', clean)
    clean = re.sub(r'【.+?:】', '', clean)
    clean = re.sub(r'\n{3,}', '\n\n', clean)
    clean = re.sub(r'[（(][^）)]{0,20}[）)]', '', clean)
    clean = re.sub(r'\*[^*]{0,20}\*', '', clean)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = re.sub(r'BREAK\s*TIME[^，。]*[，。]?', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\d+毫秒', '', clean)
    clean = re.sub(r'\s{2,}', ' ', clean).strip()
    return clean

async def generate_tts(text: str) -> str:
    """生成语音，返回 base64 字符串；优先 EdgeTTS，兜底 pyttsx3"""
    cached = _tts_cache_get(text)
    if cached:
        print(f"[TTS] 缓存命中，文字长度：{len(text)}字")
        return cached
    clean = clean_text_for_tts(text)
    print(f"[TTS] 开始生成，文字长度：{len(text)}字")
    if EDGE_TTS_OK:
        try:
            communicate = edge_tts.Communicate(
                text=clean,
                voice=VOICE_NAME,
                rate=VOICE_RATE,
                pitch=VOICE_PITCH,
                proxy=None
            )
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            if buf.tell() > 0:
                buf.seek(0)
                audio_b64 = base64.b64encode(buf.read()).decode()
                _tts_cache_set(text, audio_b64)
                print(f"[TTS] EdgeTTS 完成：{len(audio_b64)//1024}KB")
                return audio_b64
        except Exception as e:
            print(f"[TTS] EdgeTTS 失败，切换本地引擎: {e}")
    if PYTTSX3_OK:
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for v in voices:
                if 'zh' in v.id.lower() or 'chinese' in v.name.lower() or 'huihui' in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
            engine.setProperty('rate', 130)
            engine.setProperty('volume', 0.9)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp_path = f.name
            engine.save_to_file(clean, tmp_path)
            engine.runAndWait()
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            os.unlink(tmp_path)
            audio_b64 = base64.b64encode(audio_data).decode()
            _tts_cache_set(text, audio_b64)
            print(f"[TTS] pyttsx3 完成：{len(audio_b64)//1024}KB")
            return audio_b64
        except Exception as e:
            print(f"[TTS] pyttsx3 也失败: {e}")
            raise
    raise RuntimeError("TTS 引擎均不可用：请安装 edge-tts 或 pyttsx3")

# ── 启动 ──
if __name__ == '__main__':
    print("=" * 55)
    print("  [MKS] Marx Agent v2")
    print(f"  EdgeTTS: {VOICE_NAME} | rate: {VOICE_RATE} | pitch: {VOICE_PITCH}")
    if PYTTSX3_OK:
        print(f"  Local TTS: pyttsx3 (fallback)")
    print(f"  URL: http://localhost:5000")
    if WHISPER_OK:
        dev = WHISPER_DEVICE if WHISPER_DEVICE != "unknown" else "(lazy)"
        print(f"  ASR device: {dev}")
    print()
    if _is_default_key(DEEPSEEK_API_KEY):
        print("  [WARN] DeepSeek API Key not configured!")
    else:
        print("  [OK] DeepSeek API Key ready")
    if WHISPER_OK:
        print("  [OK] faster-whisper installed")
        print("  [..] Loading ASR model...")
        try:
            get_whisper_model()
        except Exception as e:
            print(f"  [WARN] ASR 加载失败: {e}")
    else:
        print("  [WARN] faster-whisper missing")
    print("  [..] Prewarming TTS cache...")
    import threading
    prewarm_thread = threading.Thread(target=_prewarm_tts, daemon=True)
    prewarm_thread.start()
    if SOCK_OK:
        print("  [OK] WebSocket ready (simple-websocket)")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
