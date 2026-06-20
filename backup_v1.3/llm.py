import requests
import json


def chat_stream(message: str, history: list, system_prompt: str,
                api_key: str, api_url: str, model: str):
    """
    流式调用 DeepSeek，yield 每个文字片段
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-(10 * 2):])
    messages.append({"role": "user", "content": message})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.75,
        "max_tokens": 400,
        "stream": True
    }

    try:
        with requests.post(api_url, headers=headers,
                           json=payload, stream=True, timeout=30) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                chunk = line[6:]
                if chunk == b"[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue
    except Exception as e:
        print(f"[LLM] 异常: {e}")
