chcp 65001
@echo off

setlocal
cd /d "%~dp0.."
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
    echo 详细步骤请查看 docs\00_快速开始.md
    echo.
    pause
    exit /b 1
)

echo 当前 Python 版本：
"%VENV_PYTHON%" --version
echo.
"%VENV_PYTHON%" -m src.app_infer
set "APP_EXIT_CODE=%ERRORLEVEL%"

if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo 推理界面异常退出，退出码：%APP_EXIT_CODE%
    echo 请查看终端中的错误信息，以及 docs\06_常见问题与排错.md
    echo 推理输出默认位于 inference_data\outputs
)

pause
exit /b %APP_EXIT_CODE%
