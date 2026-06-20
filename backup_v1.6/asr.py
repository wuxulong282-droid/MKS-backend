import numpy as np
import tempfile
import os
import io
import wave
import subprocess

WHISPER = None
FFMPEG_PATH = None


def find_ffmpeg():
    """查找本地 ffmpeg 路径"""
    global FFMPEG_PATH
    if FFMPEG_PATH:
        return FFMPEG_PATH
    known = [
        r'C:\Users\武旭龙\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe',
        r'C:\Users\武旭龙\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe',
    ]
    for p in known:
        if os.path.isfile(p):
            FFMPEG_PATH = p
            return p
    import shutil
    try:
        r = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = r.stdout.strip().split('\n')[0].strip()
            if p and os.path.isfile(p):
                FFMPEG_PATH = p
                return p
    except Exception:
        pass
    p = shutil.which('ffmpeg')
    if p and os.path.isfile(p):
        FFMPEG_PATH = p
        return p
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
    """语音识别，返回文字"""
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
