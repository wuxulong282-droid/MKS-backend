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


async def _generate_single(text: str, voice: str, rate: str, pitch: str) -> bytes:
    """单次 TTS 生成，失败抛出异常"""
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


async def _generate_with_retry(text: str, voice: str, rate: str, pitch: str, retries: int = 3) -> bytes | None:
    """带重试的 TTS 生成"""
    for attempt in range(retries):
        try:
            return await _generate_single(text, voice, rate, pitch)
        except Exception as e:
            print(f"[TTS] 尝试 {attempt+1}/{retries} 失败: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1)
    
    # 全部失败后尝试用清理过的更短文本
    short = text[:50]
    for attempt in range(2):
        try:
            return await _generate_single(short, voice, rate, pitch)
        except Exception as e:
            print(f"[TTS] 短文本尝试 {attempt+1}/2 失败: {e}")
            if attempt < 1:
                await asyncio.sleep(1)
    
    return None


def generate(text: str) -> str:
    """生成TTS，返回base64字符串，失败返回空字符串"""
    from config import VOICE_NAME, VOICE_RATE, VOICE_PITCH
    try:
        clean = clean_text(text)
        if not clean:
            return ""

        # 使用新的事件循环策略（Windows 修复）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.set_event_loop(loop)
        try:
            audio = loop.run_until_complete(
                _generate_with_retry(clean, VOICE_NAME, VOICE_RATE, VOICE_PITCH)
            )
        finally:
            loop.close()

        if audio:
            return base64.b64encode(audio).decode()

        # EdgeTTS 全部失败，尝试 pyttsx3 兜底
        return _fallback_tts(clean)
    except Exception as e:
        print(f"[TTS] 异常: {e}")
        return _fallback_tts(text)


def _fallback_tts(text: str) -> str:
    """使用 pyttsx3 离线 TTS 兜底，返回 WAV 的 base64"""
    try:
        import pyttsx3
        import tempfile
        import os

        # 截断过长文本
        text = text[:100]

        engine = pyttsx3.init(driverName='sapi5')
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 0.9)

        # 找一个中文语音
        voices = engine.getProperty('voices')
        zh_voice = None
        for v in voices:
            if 'chinese' in v.name.lower() or 'zh' in v.id.lower():
                zh_voice = v.id
                break
        if zh_voice:
            engine.setProperty('voice', zh_voice)

        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        engine.save_to_file(text, tmp.name)
        engine.runAndWait()

        with open(tmp.name, 'rb') as f:
            wav_data = f.read()
        os.unlink(tmp.name)

        b64 = base64.b64encode(wav_data).decode()
        print(f"[TTS] pyttsx3 兜底成功: {len(wav_data)} bytes")
        return b64
    except Exception as e:
        print(f"[TTS] pyttsx3 兜底也失败: {e}")
        return ""
