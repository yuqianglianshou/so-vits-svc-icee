#!/usr/bin/env python3
"""执行不依赖模型权重和 CUDA 的发布前仓库检查。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote

MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)

REQUIRED_PATHS = (
    "README.md",
    "README_en.md",
    "config_templates/config_template.json",
    "config_templates/diffusion_template.yaml",
    "src/app_train.py",
    "src/app_infer.py",
    "launchers/启动训练界面.bat",
    "launchers/启动推理界面.bat",
    "launchers/启动tensorboard.bat",
)
WINDOWS_LAUNCHERS = (
    "launchers/启动训练界面.bat",
    "launchers/启动推理界面.bat",
    "launchers/启动tensorboard.bat",
)

FORBIDDEN_SUFFIXES = (
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".wav",
    ".flac",
    ".mp3",
    ".pyc",
)
FORBIDDEN_PARTS = (".venv", ".venv311", "__pycache__")
FORBIDDEN_TEMPLATE_TEXT = (
    "svc-develop-team/so-vits-svc/discussions",
    "svc-develop-team/so-vits-svc/blob",
    "logs/44k",
)


def find_markdown_links(text: str) -> list[str]:
    """提取需要在本地文件系统验证的 Markdown 链接。"""
    links: list[str] = []
    text_without_code = FENCED_CODE_RE.sub("", text)
    for raw_target in MARKDOWN_LINK_RE.findall(text_without_code):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        links.append(unquote(target.split("#", 1)[0]))
    return links


def check_markdown_links(root: Path, markdown_files: Sequence[Path]) -> list[str]:
    """返回 Markdown 文件中所有失效的仓库内链接。"""
    errors: list[str] = []
    for path in markdown_files:
        for target in find_markdown_links(path.read_text(encoding="utf-8")):
            if not (path.parent / target).resolve().exists():
                errors.append(
                    f"{path.relative_to(root).as_posix()}: 链接目标不存在: {target}"
                )
    return errors


def check_required_paths(root: Path) -> list[str]:
    """检查发布必需的入口、配置和启动脚本。"""
    return [
        f"缺少必需路径: {relative_path}"
        for relative_path in REQUIRED_PATHS
        if not (root / relative_path).exists()
    ]


def check_json_files(root: Path, paths: Sequence[str]) -> list[str]:
    """检查 JSON 文件是否存在且能够解析。"""
    errors: list[str] = []
    for relative_path in paths:
        path = root / relative_path
        if not path.exists():
            errors.append(f"{relative_path}: 文件不存在")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append(f"{relative_path}: JSON 格式无效")
    return errors


def check_yaml_files(root: Path, paths: Sequence[str]) -> list[str]:
    """检查 YAML 文件；缺少 PyYAML 时给出可执行的安装提示。"""
    try:
        import yaml
    except ImportError:
        return ["缺少 PyYAML，无法检查 YAML；请先安装 requirements.txt"]

    errors: list[str] = []
    for relative_path in paths:
        path = root / relative_path
        if not path.exists():
            errors.append(f"{relative_path}: 文件不存在")
            continue
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            errors.append(f"{relative_path}: YAML 格式无效")
    return errors


def check_forbidden_tracked_files(tracked_files: Sequence[str]) -> list[str]:
    """拒绝把模型、生成音频、虚拟环境或缓存带入发布。"""
    errors: list[str] = []
    for value in tracked_files:
        path = Path(value)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(
            part in FORBIDDEN_PARTS for part in path.parts
        ):
            errors.append(f"Git 跟踪了发布产物或本地文件: {value}")
    return errors


def check_forbidden_text(
    root: Path,
    paths: Sequence[Path],
    forbidden_values: Sequence[str],
) -> list[str]:
    """检查面向用户的文件是否重新引入已废弃的说明。"""
    errors: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for value in forbidden_values:
            if value in text:
                errors.append(
                    f"{path.relative_to(root).as_posix()}: 包含已废弃内容: {value}"
                )
    return errors


def check_windows_launchers(root: Path, paths: Sequence[str]) -> list[str]:
    """检查 Windows 启动器是否保留必要的诊断信息和退出状态。"""
    required_fragments = {
        '"%VENV_PYTHON%" --version': "未输出 Python 版本",
        'set "APP_EXIT_CODE=%ERRORLEVEL%"': "未保存应用退出码",
        "docs\\06_常见问题与排错.md": "未指向排错文档",
        "exit /b %APP_EXIT_CODE%": "未返回应用退出码",
    }
    errors: list[str] = []
    for relative_path in paths:
        path = root / relative_path
        if not path.exists():
            errors.append(f"{relative_path}: 启动器不存在")
            continue
        text = path.read_text(encoding="utf-8")
        for fragment, message in required_fragments.items():
            if fragment not in text:
                errors.append(f"{relative_path}: {message}")
    return errors


def _git_tracked_files(root: Path) -> list[str]:
    """读取 Git 跟踪清单，Git 不可用时把原因交给调用方展示。"""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def run_checks(root: Path, tracked_files: Sequence[str] | None = None) -> list[str]:
    """运行全部轻量发布检查并汇总错误。"""
    errors = check_required_paths(root)
    markdown_files = sorted(
        path
        for base in (root / "README.md", root / "README_en.md", root / "docs")
        for path in ([base] if base.is_file() else base.rglob("*.md") if base.is_dir() else [])
    )
    errors.extend(check_markdown_links(root, markdown_files))
    errors.extend(
        check_json_files(root, ["config_templates/config_template.json"])
    )
    errors.extend(
        check_yaml_files(root, ["config_templates/diffusion_template.yaml"])
    )
    if tracked_files is None:
        try:
            tracked_files = _git_tracked_files(root)
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"无法读取 Git 跟踪文件: {exc}")
            tracked_files = []
    errors.extend(check_forbidden_tracked_files(tracked_files))
    errors.extend(check_windows_launchers(root, WINDOWS_LAUNCHERS))
    template_dir = root / ".github" / "ISSUE_TEMPLATE"
    if template_dir.exists():
        template_files = sorted(path for path in template_dir.iterdir() if path.is_file())
        errors.extend(
            check_forbidden_text(
                root,
                template_files,
                FORBIDDEN_TEMPLATE_TEXT,
            )
        )
    return errors


def main() -> int:
    """命令行入口。"""
    root = Path(__file__).resolve().parents[1]
    errors = run_checks(root)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] 发布前仓库检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
