from pathlib import Path

from scripts.check_release import (
    REQUIRED_PATHS,
    check_forbidden_tracked_files,
    check_json_files,
    check_markdown_links,
    check_required_paths,
    check_yaml_files,
    find_markdown_links,
    run_checks,
)


def test_find_markdown_links_ignores_web_anchor_and_image_links():
    """防止把外部链接、页内锚点和图片误判为仓库文件。"""
    text = (
        "[本地](docs/start.md) "
        "[网页](https://example.com) "
        "[章节](#start) "
        "![截图](images/page.png)"
    )

    assert find_markdown_links(text) == ["docs/start.md"]


def test_find_markdown_links_ignores_fenced_code_examples():
    """文档中的示例代码不能被当成真实导航链接。"""
    text = """[真实](docs/start.md)

```python
example = "[示例](docs/missing.md)"
```
"""

    assert find_markdown_links(text) == ["docs/start.md"]


def test_check_markdown_links_reports_missing_target(tmp_path: Path):
    """链接目标被删除时，检查结果必须指出来源文件和目标。"""
    readme = tmp_path / "README.md"
    readme.write_text("[不存在](docs/missing.md)", encoding="utf-8")

    assert check_markdown_links(tmp_path, [readme]) == [
        "README.md: 链接目标不存在: docs/missing.md"
    ]


def test_check_required_paths_reports_every_missing_path(tmp_path: Path):
    """发布所需入口缺失时，检查器应一次报告全部问题。"""
    errors = check_required_paths(tmp_path)

    assert len(errors) == len(REQUIRED_PATHS)
    assert "缺少必需路径: README.md" in errors


def test_check_json_files_reports_invalid_json(tmp_path: Path):
    """损坏的配置模板不能进入发布版本。"""
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    assert check_json_files(tmp_path, ["broken.json"]) == [
        "broken.json: JSON 格式无效"
    ]


def test_check_yaml_files_reports_invalid_yaml(tmp_path: Path):
    """损坏的扩散配置模板不能进入发布版本。"""
    path = tmp_path / "broken.yaml"
    path.write_text("train: [", encoding="utf-8")

    assert check_yaml_files(tmp_path, ["broken.yaml"]) == [
        "broken.yaml: YAML 格式无效"
    ]


def test_check_forbidden_tracked_files_rejects_generated_assets():
    """模型、输出音频和缓存文件不能被 Git 跟踪。"""
    tracked_files = [
        "model_assets/workspaces/demo/G_100.pth",
        "inference_data/outputs/demo.wav",
        "src/__pycache__/app.cpython-311.pyc",
        "README.md",
    ]

    assert check_forbidden_tracked_files(tracked_files) == [
        "Git 跟踪了发布产物或本地文件: model_assets/workspaces/demo/G_100.pth",
        "Git 跟踪了发布产物或本地文件: inference_data/outputs/demo.wav",
        "Git 跟踪了发布产物或本地文件: src/__pycache__/app.cpython-311.pyc",
    ]


def test_run_checks_accepts_minimal_valid_repository(tmp_path: Path):
    """满足发布契约的最小仓库应通过全部检查。"""
    for relative_path in REQUIRED_PATHS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "{}" if path.suffix == ".json" else ""
        path.write_text(content, encoding="utf-8")

    assert run_checks(tmp_path, tracked_files=[]) == []
