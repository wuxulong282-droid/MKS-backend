@echo off
echo 安装所有依赖...
pip install -r requirements.txt ^
 -i https://mirrors.aliyun.com/pypi/simple/ ^
 --timeout 120
pause
