# MKS 版本记录

## v1.1 — 2026-06-01 19:40
RealtimeSTT → faster-whisper 重构 + CUDA 正常 + 30秒录音限制

### 变更文件
- `asr.py`：替换为直接调用 faster-whisper，去掉 AudioToTextRecorder 的多进程问题
- `app.py`：`get_recorder()` → `get_model()`；修复 `/health` 路由装饰器缺失
- `frontend/index.html`：按住说话增加 30 秒自动停止

### 状态
- [x] 语音识别 /chat_voice 返回 200
- [x] 持续按住说话不会无限计时（30秒限制）
- [x] 文字对话可用（DeepSeek API）
- [x] TTS 可用
- [x] 服务通过 Waitress 启动正常

### 备份位置
`backup_v1.1/` — asr.py, app.py, index.html, config.py, llm.py, tts.py

### 回退方法
```powershell
Copy-Item "MKS\backup_v1.1\app.py" "MKS\app.py" -Force
Copy-Item "MKS\backup_v1.1\asr.py" "MKS\asr.py" -Force
Copy-Item "MKS\backup_v1.1\index.html" "MKS\frontend\index.html" -Force
```
