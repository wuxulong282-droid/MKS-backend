import os

# ASR 模块 — 云端使用浏览器 SpeechRecognition，此模块仅保留占位
_HAS_ASR = bool(os.getenv('XFYUN_APPID'))

def get_model():
    return None

def transcribe(audio_bytes: bytes) -> str:
    """语音识别 — 云端使用浏览器 SpeechRecognition，后端不处理"""
    return ""

def transcribe_xfyun(audio_bytes: bytes) -> str:
    return ""
