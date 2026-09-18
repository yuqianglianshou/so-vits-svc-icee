from pathlib import Path

from scripts.check_release import check_windows_launchers


def _write_launcher(root: Path, name: str, content: str) -> None:
    """在临时仓库中创建一个 Windows 启动器。"""
    launcher_dir = root / "launchers"
    launcher_dir.mkdir(exist_ok=True)
    (launcher_dir / name).write_text(content, encoding="utf-8")


def test_launchers_require_diagnostics_and_exit_code(tmp_path: Path) -> None:
    _write_launcher(
        tmp_path,
        "broken.bat",
        '"%VENV_PYTHON%" -m src.app_train\npause\n',
    )

    errors = check_windows_launchers(tmp_path, ["launchers/broken.bat"])

    assert any("Python 版本" in error for error in errors)
    assert any("退出码" in error for error in errors)
    assert any("排错文档" in error for error in errors)


def test_launchers_accept_complete_failure_handling(tmp_path: Path) -> None:
    _write_launcher(
        tmp_path,
        "complete.bat",
        "\n".join(
            (
                '"%VENV_PYTHON%" --version',
                '"%VENV_PYTHON%" -m src.app_train',
                'set "APP_EXIT_CODE=%ERRORLEVEL%"',
                'echo 请查看 docs\\06_常见问题与排错.md',
                'exit /b %APP_EXIT_CODE%',
            )
        ),
    )

    assert check_windows_launchers(tmp_path, ["launchers/complete.bat"]) == []
