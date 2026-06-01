import sys
import os

print("Python路径:", sys.executable)
print("Python版本:", sys.version)
print("")

libs = ["flask", "flask_cors", "flask_sock", "edge_tts",
 "whisper", "webrtcvad", "requests", "numpy"]

for lib in libs:
    try:
        __import__(lib)
        print("\u2713 " + lib + " 已安装")
    except ImportError:
        print("\u2717 " + lib + " 未安装")
