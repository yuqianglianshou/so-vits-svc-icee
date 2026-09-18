chcp 65001
@echo off
setlocal
cd /d "%~dp0.."
set "VENV_PYTHON=.venv311\Scripts\python.exe"

echo 正在启动 TensorBoard...
echo 如果看到输出了一条网址（大概率是 localhost:6006）就可以访问该网址进入 TensorBoard
echo 当前会监控整个 model_assets\workspaces 目录，适配按模型名分开的训练产物。

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
"%VENV_PYTHON%" -m tensorboard.main --logdir=model_assets/workspaces
set "APP_EXIT_CODE=%ERRORLEVEL%"

if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo TensorBoard 异常退出，退出码：%APP_EXIT_CODE%
    echo 请查看终端中的错误信息，以及 docs\06_常见问题与排错.md
    echo 训练产物默认位于 model_assets\workspaces
)

pause
exit /b %APP_EXIT_CODE%
