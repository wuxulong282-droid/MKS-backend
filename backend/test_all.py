import requests
import json

BASE = "http://localhost:5000"

# 测试1：基础连通
try:
 r = requests.get(f"{BASE}/health", timeout=5)
 print("health:", r.json())
except Exception as e:
 print("health失败:", e)

# 测试2：文字对话
try:
 r = requests.post(f"{BASE}/chat",
 json={"message": "你好", "history": [], "tts": False},
 timeout=15)
 print("chat状态码:", r.status_code)
 print("chat返回:", r.text[:200])
except Exception as e:
 print("chat失败:", e)

# 测试3：TTS
try:
 r = requests.post(f"{BASE}/tts",
 json={"text": "测试语音"},
 timeout=15)
 print("tts状态码:", r.status_code)
 print("tts返回长度:", len(r.text))
except Exception as e:
 print("tts失败:", e)
