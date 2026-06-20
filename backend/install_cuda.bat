@echo off
chcp 65001 >nul
echo.
echo ============================================
echo   MKS - CUDA PyTorch 安装脚本
echo ============================================
echo.
echo 正在安装 CUDA 12.1 版本的 PyTorch 2.5.1...
echo.
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --timeout 600
echo.
echo ============================================
echo  验证 CUDA 是否可用...
echo ============================================
python -c "import torch; print('torch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
echo.
if %errorlevel% equ 0 (
    echo ✅ CUDA PyTorch 安装成功！
) else (
    echo ❌ 安装失败，请检查错误信息。
)
pause
