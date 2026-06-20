import numpy as np
import tempfile
import os
import io
import wave
import subprocess
import hashlib
import hmac
import base64
import json
import time
from datetime import datetime
import websocket
import threading

WHISPER = None
FFMPEG_PATH = None


def find_ffmpeg():
    """查找本地 ffmpeg 路径"""
    global FFMPEG_PATH
    if FFMPEG_PATH:
        return FFMPEG_PATH
    import shutil
    # 1) shutil.which — 跨平台最通用
    p = shutil.which('ffmpeg')
    if p and os.path.isfile(p):
        print(f'[ASR] ffmpeg found via shutil: {p}')
        FFMPEG_PATH = p
        return p
    # 2) subprocess which — Linux 环境
    import subprocess as _sp
    try:
        r = _sp.run(['which', 'ffmpeg'], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = r.stdout.strip()
            if p and os.path.isfile(p):
                print(f'[ASR] ffmpeg found via which: {p}')
                FFMPEG_PATH = p
                return p
    except Exception:
        pass
    # 3) Windows where
    try:
        r = _sp.run(['where', 'ffmpeg'], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = r.stdout.strip().split('\n')[0].strip()
            if p and os.path.isfile(p):
                FFMPEG_PATH = p
                return p
    except Exception:
        pass
    # 4) Windows 硬编码路径（本地开发）
    win_known = [
        r'C:\Users\武旭龙\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe',
        r'C:\Users\武旭龙\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe',
    ]
    for p in win_known:
        if os.path.isfile(p):
            FFMPEG_PATH = p
            return p
    # 5) 常见 Linux 路径
    linux_candidates = [
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/nix/var/nix/profiles/default/bin/ffmpeg',
        '/root/.nix-profile/bin/ffmpeg',
        '/opt/homebrew/bin/ffmpeg',
    ]
    for c in linux_candidates:
        if os.path.isfile(c):
            print(f'[ASR] ffmpeg found linux: {c}')
            FFMPEG_PATH = c
            return c
    # 6) 递归搜索 PATH 里的 ffmpeg
    try:
        path_dirs = os.environ.get('PATH', '').split(':')
        for d in path_dirs:
            for name in ['ffmpeg', 'ffmpeg.exe']:
                fp = os.path.join(d, name)
                if os.path.isfile(fp):
                    print(f'[ASR] ffmpeg found in PATH: {fp}')
                    FFMPEG_PATH = fp
                    return fp
    except Exception:
        pass
    print('[ASR] ffmpeg 未找到')
    return None


def decode_audio_to_wav(audio_bytes: bytes) -> bytes:
    """把前端传来的 webm 转为 16kHz 16-bit WAV PCM bytes"""
    ff = find_ffmpeg()
    if not ff:
        print("[ASR] ffmpeg 未找到")
        return None

    suffix = '.webm'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        in_path = f.name
    out_path = in_path + '.wav'

    try:
        r = subprocess.run(
            [ff, '-y', '-i', in_path, '-ar', '16000', '-ac', '1', '-f', 'wav', out_path],
            capture_output=True, timeout=15
        )
        if r.returncode != 0:
            print(f"[ASR] ffmpeg失败: {r.stderr.decode(errors='ignore')[:100]}")
            return None
        with open(out_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"[ASR] decode异常: {e}")
        return None
    finally:
        for p in [in_path, out_path]:
            try:
                os.unlink(p)
            except:
                pass


def get_model():
    """预先加载 whisper 模型（避免 multiprocessing 冲突）"""
    global WHISPER
    if WHISPER is None:
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        print(f"[ASR] 加载 faster-whisper small, device={device}, compute={compute}")
        WHISPER = WhisperModel("small", device=device, compute_type=compute)
        print("[ASR] 加载完成")
    return WHISPER


def transcribe(audio_bytes: bytes) -> str:
    """语音识别，返回文字 — 优先用讯飞云端ASR，兜底用本地Whisper"""
    # 优先用讯飞云端ASR
    if os.getenv('XFYUN_APPID'):
        text = transcribe_xfyun(audio_bytes)
        if text:
            return text
        print('[ASR] 讯飞识别为空，尝试本地Whisper')

    # 兜底用本地Whisper（云端没有 faster-whisper）
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print('[ASR] faster-whisper 未安装，跳过本地Whisper')
        return ''
    try:
        wav_data = decode_audio_to_wav(audio_bytes)
        if wav_data is None or len(wav_data) < 5000:
            print("[ASR] 音频太短或解码失败")
            return ""

        # 从 wav bytes 解析为 float32 numpy 数组
        with wave.open(io.BytesIO(wav_data), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        model = get_model()
        segments, _ = model.transcribe(
            samples, language="zh", beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        text = "".join(s.text for s in segments).strip()
        if not text:
            text = ""
        print(f"[ASR] 识别: '{text}'")
        return text
    except Exception as e:
        print(f"[ASR] 异常: {e}")
        import traceback
        traceback.print_exc()
        return ""


def transcribe_xfyun(audio_bytes: bytes) -> str:
    """
    用讯飞云端API识别音频，返回文字
    audio_bytes: 任意格式音频
    """
    APPID = os.getenv('XFYUN_APPID','')
    APIKEY = os.getenv('XFYUN_APIKEY','')
    APISECRET = os.getenv('XFYUN_APISECRET','')

    if not APPID:
        print('[ASR] 讯飞配置缺失，跳过')
        return ''

    # 用ffmpeg转为PCM格式（16kHz单声道16bit）
    ff = find_ffmpeg()
    if not ff:
        print('[ASR] ffmpeg未找到，无法转码给讯飞')
        return ''

    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
        f.write(audio_bytes)
        in_path = f.name
    out_path = in_path + '.pcm'

    try:
        r = subprocess.run([
            ff, '-y', '-i', in_path,
            '-ar', '16000', '-ac', '1',
            '-f', 's16le', out_path
        ], capture_output=True, timeout=15)
        if r.returncode != 0:
            print(f'[ASR] ffmpeg失败')
            return ''
        with open(out_path, 'rb') as f:
            pcm_data = f.read()
    except Exception as e:
        print(f'[ASR] 转码异常: {e}')
        return ''
    finally:
        for p in [in_path, out_path]:
            try: os.unlink(p)
            except: pass

    # 生成鉴权URL
    def create_url():
        url = 'wss://iat-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = now.strftime('%a, %d %b %Y %H:%M:%S GMT')
        signature_origin = f'host: iat-api.xfyun.cn\ndate: {date}\nGET /v2/iat HTTP/1.1'
        signature_sha = hmac.new(
            APISECRET.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature = base64.b64encode(signature_sha).decode('utf-8')
        auth_str = f'api_key="{APIKEY}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        auth = base64.b64encode(auth_str.encode()).decode()
        return f'{url}?authorization={auth}&date={date}&host=iat-api.xfyun.cn'

    result_text = []
    done_event = threading.Event()

    def on_message(ws, message):
        data = json.loads(message)
        code = data.get('code', -1)
        if code != 0:
            print(f'[ASR] 讯飞错误: {code}')
            done_event.set()
            return
        cws_data = data.get('data', {})
        result = cws_data.get('result', {})
        ws_data = result.get('ws', [])
        for w in ws_data:
            for cw in w.get('cw', []):
                result_text.append(cw.get('w', ''))
        status = cws_data.get('status', 0)
        if status == 2:
            done_event.set()

    def on_open(ws):
        # 发送参数帧
        params = {
            'common': {'app_id': APPID},
            'business': {
                'language': 'zh_cn',
                'domain': 'iat',
                'accent': 'mandarin',
                'vad_eos': 3000,
                'dwa': 'wpgs',
            },
            'data': {
                'status': 0,
                'format': 'audio/L16;rate=16000',
                'encoding': 'raw',
                'audio': base64.b64encode(pcm_data[:1280]).decode()
            }
        }
        ws.send(json.dumps(params))

        # 分片发送剩余音频
        chunk_size = 1280
        offset = 1280
        while offset < len(pcm_data):
            chunk = pcm_data[offset:offset+chunk_size]
            ws.send(json.dumps({
                'data': {
                    'status': 1,
                    'format': 'audio/L16;rate=16000',
                    'encoding': 'raw',
                    'audio': base64.b64encode(chunk).decode()
                }
            }))
            offset += chunk_size
            time.sleep(0.04)

        # 发送结束帧
        ws.send(json.dumps({
            'data': {
                'status': 2,
                'format': 'audio/L16;rate=16000',
                'encoding': 'raw',
                'audio': ''
            }
        }))

    def on_error(ws, error):
        print(f'[ASR] WebSocket错误: {error}')
        done_event.set()

    def on_close(ws, *args):
        done_event.set()

    try:
        ws = websocket.WebSocketApp(
            create_url(),
            on_message=on_message,
            on_open=on_open,
            on_error=on_error,
            on_close=on_close
        )
        t = threading.Thread(target=ws.run_forever)
        t.daemon = True
        t.start()
        done_event.wait(timeout=15)
        result = ''.join(result_text).strip()
        print(f'[ASR] 讯飞识别结果: "{result}"')
        return result
    except Exception as e:
        print(f'[ASR] 识别异常: {e}')
        return ''
