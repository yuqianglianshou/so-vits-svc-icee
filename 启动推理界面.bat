chcp 65001
@echo off

setlocal
cd /d "%~dp0"
set "VENV_PYTHON=.venv311\Scripts\python.exe"

echo 初始化并启动歌声转换页面……初次启动可能会花上较长时间
echo 运行过程中请勿关闭此窗口！

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

"%VENV_PYTHON%" app_infer.py

pause
