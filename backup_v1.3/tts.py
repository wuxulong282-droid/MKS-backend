import asyncio
import base64
import io
import re


def clean_text(text: str) -> str:
    """清理不适合朗读的内容"""
    text = re.sub(r'\*+(.+?)\*+', r'\1', text)
    text = re.sub(r'[（(][^）)]{0,30}[）)]', '', text)
    text = re.sub(r'【[^】]{0,10}:】', '', text)
    text = re.sub(r'BREAK\s*TIME[^，。]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+毫秒', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


async def _generate(text: str, voice: str, rate: str, pitch: str) -> bytes:
    import edge_tts
    communicate = edge_tts.Communicate(
        text=text, voice=voice, rate=rate, pitch=pitch
    )
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


def generate(text: str) -> str:
    """生成TTS，返回base64字符串，失败返回空字符串"""
    from config import VOICE_NAME, VOICE_RATE, VOICE_PITCH
    try:
        clean = clean_text(text)
        if not clean:
            return ""
        audio = asyncio.run(_generate(clean, VOICE_NAME, VOICE_RATE, VOICE_PITCH))
        return base64.b64encode(audio).decode()
    except Exception as e:
        print(f"[TTS] 异常: {e}")
        return ""
