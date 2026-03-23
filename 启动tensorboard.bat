chcp 65001
@echo off
setlocal
cd /d "%~dp0"
set "VENV_PYTHON=.venv311\Scripts\python.exe"

echo 正在启动 TensorBoard...
echo 如果看到输出了一条网址（大概率是 localhost:6006）就可以访问该网址进入 TensorBoard
echo 当前会监控整个 logs 目录，适配按模型名分开的训练产物。

if not exist "%VENV_PYTHON%" (
    echo.
    echo 未检测到项目根目录下的 .venv311 虚拟环境。
    echo 请先在项目根目录执行：
    echo   python3.11 -m venv .venv311
    echo   .venv311\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m tensorboard.main --logdir=logs

pause
