# 发布稳定化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变模型行为、默认训练参数和现有目录契约的前提下，将仓库整理为可公开发布、可自助使用、可自动检查的稳定发布候选版本。

**Architecture:** 中文 README 作为项目事实入口，英文 README 同步关键事实，详细内容下沉到按用户任务组织的 docs。新增只依赖轻量开发环境的发布检查和单元测试，并让 GitHub CI、Issue 模板、启动脚本与当前项目实际入口保持一致。

**Tech Stack:** Python 3.11、Markdown、Windows Batch、GitHub Actions、Ruff、pytest、现有 PyYAML。

**Spec:** `docs/superpowers/specs/2026-09-18-release-stabilization-design.md`

## Global Constraints

- 不修改模型结构、loss、训练参数和推理参数默认值。
- 不改变 checkpoint、训练特征包、推理包和模型工作区格式。
- 不改变训练与推理运行时数据流。
- 不引入新的生产运行依赖；Ruff 和 pytest 仅进入开发依赖。
- 不宣称未经实际验证的平台、显卡、CUDA 组合或性能数据。
- 所有新增代码提供必要的中文注释。
- 每个任务独立提交，禁止夹带无关格式化或重构。

---

### Task 1: 建立发布检查测试骨架

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/test_release_checks.py`
- Create: `scripts/check_release.py`

**Interfaces:**
- Produces: `find_markdown_links(text: str) -> list[str]`
- Produces: `check_markdown_links(root: Path, markdown_files: Sequence[Path]) -> list[str]`
- Produces: `check_required_paths(root: Path) -> list[str]`
- Produces: `check_json_files(root: Path, paths: Sequence[str]) -> list[str]`
- Produces: `check_forbidden_tracked_files(root: Path, tracked_files: Sequence[str]) -> list[str]`
- Produces: `run_checks(root: Path, tracked_files: Sequence[str] | None = None) -> list[str]`
- Produces: CLI exit code `0` when all checks pass and `1` when any check fails.

- [ ] **Step 1: 添加开发依赖文件**

```text
pytest==8.3.5
ruff==0.11.2
```

保持生产 `requirements.txt` 不变。

- [ ] **Step 2: 编写失败的相对链接解析测试**

```python
from pathlib import Path

from scripts.check_release import check_markdown_links, find_markdown_links


def test_find_markdown_links_ignores_web_and_anchor_links():
    text = "[本地](docs/start.md) [网页](https://example.com) [章节](#start)"
    assert find_markdown_links(text) == ["docs/start.md"]


def test_check_markdown_links_reports_missing_target(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("[不存在](docs/missing.md)", encoding="utf-8")
    assert check_markdown_links(tmp_path, [readme]) == [
        "README.md: 链接目标不存在: docs/missing.md"
    ]
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_release_checks.py -v`

Expected: FAIL，错误包含 `No module named 'scripts.check_release'`。

- [ ] **Step 4: 实现 Markdown 链接解析和检查**

在 `scripts/check_release.py` 中实现：

```python
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote


MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def find_markdown_links(text: str) -> list[str]:
    links: list[str] = []
    for raw_target in MARKDOWN_LINK_RE.findall(text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        links.append(unquote(target.split("#", 1)[0]))
    return links


def check_markdown_links(root: Path, markdown_files: Sequence[Path]) -> list[str]:
    errors: list[str] = []
    for path in markdown_files:
        for target in find_markdown_links(path.read_text(encoding="utf-8")):
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(
                    f"{path.relative_to(root).as_posix()}: 链接目标不存在: {target}"
                )
    return errors
```

- [ ] **Step 5: 增加必需路径、JSON 和禁止跟踪文件测试**

覆盖以下行为：

```python
def test_check_required_paths_reports_every_missing_path(tmp_path: Path):
    errors = check_required_paths(tmp_path)
    assert len(errors) == len(REQUIRED_PATHS)
    assert "缺少必需路径: README.md" in errors


def test_check_json_files_reports_invalid_json(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    assert check_json_files(tmp_path, ["broken.json"]) == [
        "broken.json: JSON 格式无效"
    ]


def test_check_forbidden_tracked_files_rejects_model_and_audio_files(tmp_path: Path):
    errors = check_forbidden_tracked_files(
        tmp_path,
        ["model_assets/workspaces/demo/G_100.pth", "inference_data/outputs/demo.wav"],
    )
    assert errors == [
        "Git 跟踪了发布产物或本地文件: model_assets/workspaces/demo/G_100.pth",
        "Git 跟踪了发布产物或本地文件: inference_data/outputs/demo.wav",
    ]


def test_run_checks_accepts_minimal_valid_repository(tmp_path: Path):
    for relative_path in REQUIRED_PATHS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "{}" if path.suffix == ".json" else ""
        path.write_text(content, encoding="utf-8")
    assert run_checks(tmp_path, tracked_files=[]) == []
```

测试仓库最小结构包含：

```text
README.md
README_en.md
config_templates/config_template.json
config_templates/diffusion_template.yaml
src/app_train.py
src/app_infer.py
launchers/启动训练界面.bat
launchers/启动推理界面.bat
launchers/启动tensorboard.bat
```

- [ ] **Step 6: 运行新增测试确认失败**

Run: `python -m pytest tests/test_release_checks.py -v`

Expected: FAIL，缺少 `check_required_paths`、`check_json_files`、`check_forbidden_tracked_files` 和 `run_checks`。

- [ ] **Step 7: 实现剩余检查及 CLI**

实现规则：

```python
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

FORBIDDEN_SUFFIXES = (".pth", ".pt", ".ckpt", ".onnx", ".wav", ".flac", ".mp3", ".pyc")
FORBIDDEN_PARTS = (".venv", ".venv311", "__pycache__")
```

`run_checks()` 汇总全部错误，不提前中止。`main()` 输出 `[OK]` 或逐条 `[FAIL]`，并返回明确退出码。YAML 解析仅在 PyYAML 可用时执行；缺少 PyYAML 时报告如何安装项目依赖。

- [ ] **Step 8: 运行测试和真实仓库检查**

Run: `python -m pytest tests/test_release_checks.py -v`

Expected: PASS。

Run: `python scripts/check_release.py`

Expected: 当前仓库可能因已知文档死链或上游链接返回 FAIL；记录失败项供 Task 2 和 Task 4 修复，不放宽规则。

- [ ] **Step 9: 提交检查骨架**

```bash
git add requirements-dev.txt scripts/check_release.py tests/test_release_checks.py
git commit -m "test: 添加发布前仓库检查"
```

---

### Task 2: 重写项目首页和文档导航

**Files:**
- Modify: `README.md`
- Modify: `README_en.md`
- Modify: `docs/README.md`
- Create: `docs/00_快速开始.md`
- Create: `docs/07_环境与兼容性.md`
- Create: `docs/08_模型依赖清单.md`
- Create: `docs/09_已知限制.md`
- Create: `docs/90_开发与发布检查.md`

**Interfaces:**
- Consumes: `scripts/check_release.py` 的 Markdown 链接检查。
- Produces: 中文 README 为项目事实入口；英文 README 同步关键事实；docs 为详细说明入口。

- [ ] **Step 1: 为新文档导航添加失败测试**

在 `tests/test_release_checks.py` 增加：

```python
def test_repository_docs_index_has_no_missing_local_links():
    root = Path(__file__).resolve().parents[1]
    files = [root / "README.md", root / "README_en.md", root / "docs" / "README.md"]
    assert check_markdown_links(root, files) == []
```

- [ ] **Step 2: 运行测试确认现有死链被发现**

Run: `python -m pytest tests/test_release_checks.py::test_repository_docs_index_has_no_missing_local_links -v`

Expected: FAIL，至少报告 `docs/90_项目现状与维护建议.md` 不存在。

- [ ] **Step 3: 重写中文 README**

按设计文档第 5.1 节顺序编写，必须包含：

- “本地离线的 So-VITS-SVC 可视化训练与歌声转换工作台”定位。
- 相对上游的六项差异表。
- `Windows 10/11 + NVIDIA GPU + Python 3.11` 主支持范围。
- `.venv311` 创建、PyTorch 安装提示、`requirements.txt` 安装。
- `python -m src.app_train` 与 `python -m src.app_infer`。
- 训练 1～6 步简表，明确扩散和索引的可选性质以实际 UI 为准。
- 新增文档入口和已知限制入口。
- 不承诺最低显存、Linux 完整支持或模型效果提升倍数。

- [ ] **Step 4: 同步英文 README**

英文版同步定位、差异、环境、安装、启动、训练/推理路线、资产和限制；增加说明：详细用户文档当前以简体中文为主。

- [ ] **Step 5: 重写 docs 索引**

删除不存在的维护建议链接，按以下分组导航：

```text
快速开始
日常使用
理解模型
问题排查
开发与研究
```

- [ ] **Step 6: 编写五份发布型文档**

要求：

- `00` 只描述首次成功路径，并把异常链接到 `02`。
- `07` 只写已验证或明确标注待人工验证的环境事实。
- `08` 集中列出 ContentVec、RMVPE、G_0、D_0、model_0、NSF-HiFiGAN 的用途和路径。
- `09` 明确平台、单说话人、实时转换、ONNX、模型效果和授权限制。
- `90` 写明本计划中的检查命令、人工 Windows 验收表和 Release Candidate 规则。

- [ ] **Step 7: 运行文档检查**

Run: `python -m pytest tests/test_release_checks.py::test_repository_docs_index_has_no_missing_local_links -v`

Expected: PASS。

Run: `python scripts/check_release.py`

Expected: 文档链接相关错误全部消失；GitHub 模板相关错误留给 Task 4。

- [ ] **Step 8: 提交首页和导航**

```bash
git add README.md README_en.md docs/README.md docs/00_快速开始.md docs/07_环境与兼容性.md docs/08_模型依赖清单.md docs/09_已知限制.md docs/90_开发与发布检查.md tests/test_release_checks.py
git commit -m "docs: 重构发布首页与文档导航"
```

---

### Task 3: 去重并校准现有用户文档

**Files:**
- Modify: `docs/01_Windows安装训练推理指南.md`
- Modify: `docs/02_Windows常见问题排查.md`
- Modify: `docs/03_核心模型与底模说明.md`
- Modify: `docs/04_训练流程与原理说明.md`
- Modify: `docs/05_音质配置推荐.md`
- Modify: `docs/06_训练术语与参数速查.md`
- Modify: `tests/test_release_checks.py`

**Interfaces:**
- Consumes: Task 2 新增的环境、资产和限制文档。
- Produces: 用户操作、概念、调优和排错各自只有一个详细事实来源。

- [ ] **Step 1: 增加文档旧路径扫描测试**

```python
def test_user_docs_do_not_reference_removed_runtime_paths():
    root = Path(__file__).resolve().parents[1]
    text = "\n".join(path.read_text(encoding="utf-8") for path in (root / "docs").glob("*.md"))
    forbidden = ("logs/44k", "pretrain/", "python webUI.py", "python inference_main.py")
    assert [value for value in forbidden if value in text] == []
```

允许在研究文档的明确“旧版配置示例”中引用旧值时，使用具体文件白名单，不对整个 docs 放宽规则。

- [ ] **Step 2: 运行测试确认失败项**

Run: `python -m pytest tests/test_release_checks.py::test_user_docs_do_not_reference_removed_runtime_paths -v`

Expected: 如存在旧路径则 FAIL；若当前已无旧路径，保留该回归测试并继续下一步。

- [ ] **Step 3: 收敛安装指南**

`01` 保留完整操作流程，但将环境矩阵、模型资产详细表和故障解释链接到 `07`、`08`、`02`，避免重复维护下载细节。

- [ ] **Step 4: 收敛故障排查**

`02` 每个问题统一为：

```text
现象
可能原因
检查方法
解决步骤
仍未解决时需要提供的信息
```

命令和目录与当前工作区结构一致。

- [ ] **Step 5: 校准模型与训练文档**

- `03` 只解释组件和资产职责。
- `04` 只解释训练 1～6 步的数据流和代码对应。
- `05` 明确推荐配置是经验基线，不宣称客观最优。
- `06` 只作为参数与术语速查，不重复完整训练教程。

- [ ] **Step 6: 运行所有文档检查**

Run: `python -m pytest tests/test_release_checks.py -v`

Expected: PASS。

Run: `python scripts/check_release.py`

Expected: 用户文档相关检查全部通过。

- [ ] **Step 7: 提交文档校准**

```bash
git add docs/01_Windows安装训练推理指南.md docs/02_Windows常见问题排查.md docs/03_核心模型与底模说明.md docs/04_训练流程与原理说明.md docs/05_音质配置推荐.md docs/06_训练术语与参数速查.md tests/test_release_checks.py
git commit -m "docs: 校准用户指南与排错内容"
```

---

### Task 4: 更新 GitHub 项目协作入口

**Files:**
- Create: `CHANGELOG.md`
- Create: `CONTRIBUTING.md`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Replace: `.github/ISSUE_TEMPLATE/bug_report.yaml`
- Replace: `.github/ISSUE_TEMPLATE/ask_for_help.yaml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yaml`
- Modify: `.github/ISSUE_TEMPLATE/config.yml`
- Delete: `.github/ISSUE_TEMPLATE/default.md`
- Delete: `.github/ISSUE_TEMPLATE/bug_report_en_US.yaml`
- Delete: `.github/ISSUE_TEMPLATE/ask_for_help_en_US.yaml`
- Modify: `tests/test_release_checks.py`

**Interfaces:**
- Produces: 当前项目的 Bug、帮助、功能建议和 PR 信息入口。
- Consumes: README、开发检查文档和当前仓库 URL。

- [ ] **Step 1: 添加上游链接回归测试**

```python
def test_github_templates_do_not_point_users_to_upstream_support():
    root = Path(__file__).resolve().parents[1]
    template_dir = root / ".github" / "ISSUE_TEMPLATE"
    text = "\n".join(path.read_text(encoding="utf-8") for path in template_dir.glob("*") if path.is_file())
    assert "svc-develop-team/so-vits-svc/issues" not in text
    assert "svc-develop-team/so-vits-svc/discussions" not in text
    assert "logs/44k" not in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_release_checks.py::test_github_templates_do_not_point_users_to_upstream_support -v`

Expected: FAIL，报告上游 Discussions 或旧目录。

- [ ] **Step 3: 重写中文 Issue 模板**

Bug 模板收集：版本、系统、GPU、Python/PyTorch/CUDA、入口、工作区状态、复现步骤、预期、实际、日志和复现稳定性。

帮助模板收集：已阅读的文档、环境、所处步骤、页面提示、日志和已经尝试的处理。

功能建议模板收集：用户问题、期望流程、替代方式、兼容性影响；不得预设实现方案。

- [ ] **Step 4: 收敛模板数量和讨论入口**

删除继承自上游且不再维护的英文重复模板和默认空模板。若当前仓库 Discussions 未确认启用，`config.yml` 不提供外部 contact link，并保持 `blank_issues_enabled: false`。

- [ ] **Step 5: 添加发布与贡献文档**

- `CHANGELOG.md` 使用 Keep a Changelog 风格，建立“未发布”段并记录可视化训练、推理、工作区、依赖管理和文档改进。
- `CONTRIBUTING.md` 写明 Python 3.11、开发依赖安装、分支、提交范围和完整检查命令。
- PR 模板包含改动摘要、范围、验证命令、模型兼容性、文档更新和截图项。

- [ ] **Step 6: 验证模板**

Run: `python -m pytest tests/test_release_checks.py -v`

Expected: PASS。

Run: `python scripts/check_release.py`

Expected: 不再报告上游支持链接和旧目录。

- [ ] **Step 7: 提交协作入口**

```bash
git add CHANGELOG.md CONTRIBUTING.md .github tests/test_release_checks.py
git commit -m "docs: 更新项目协作与发布入口"
```

---

### Task 5: 增加不依赖模型的业务单元测试

**Files:**
- Create: `tests/test_path_utils.py`
- Create: `tests/test_train_ui_config_sync.py`
- Create: `tests/test_train_ui_workspace.py`
- Create: `tests/test_infer_ui_files.py`
- Modify only if a test exposes a release-blocking defect: corresponding focused module under `src/`

**Interfaces:**
- Tests existing public functions without changing their signatures.
- Uses `tmp_path` and monkeypatching; never writes to real workspaces.

- [ ] **Step 1: 为路径工具编写测试**

覆盖：

```python
from src import path_utils


def test_dependency_paths_are_under_model_assets():
    dependency_root = path_utils.ROOT / "model_assets" / "dependencies"
    paths = (
        path_utils.get_contentvec_hf_path(),
        path_utils.get_rmvpe_path(),
        path_utils.get_nsf_hifigan_model_path(),
        path_utils.get_sovits_g0_path(),
        path_utils.get_sovits_d0_path(),
        path_utils.get_diffusion_model_0_path(),
    )
    assert all(dependency_root in path.parents for path in paths)


def test_runtime_dependency_mapping_preserves_expected_names(tmp_path, monkeypatch):
    monkeypatch.setattr(path_utils, "ROOT", tmp_path)
    monkeypatch.setattr(path_utils, "BASE_MODEL_44K_DIR", tmp_path / "deps" / "44k")
    monkeypatch.setattr(
        path_utils,
        "BASE_MODEL_44K_DIFFUSION_DIR",
        tmp_path / "deps" / "44k" / "diffusion",
    )
    for source in (
        path_utils.get_sovits_g0_path(),
        path_utils.get_sovits_d0_path(),
        path_utils.get_diffusion_model_0_path(),
    ):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"model")
    path_utils.ensure_runtime_base_models("demo")
    assert (tmp_path / "model_assets/workspaces/demo/G_0.pth").exists()
    assert (tmp_path / "model_assets/workspaces/demo/D_0.pth").exists()
    assert (tmp_path / "model_assets/workspaces/demo/diffusion/model_0.pt").exists()
```

断言 G_0、D_0、model_0、ContentVec 和 vocoder 路径与文档资产表一致。

- [ ] **Step 2: 为 batch size 配置同步编写测试**

覆盖：

```python
from src.train_ui import config_sync


def test_load_template_batch_size_reads_train_value(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"train": {"batch_size": 7}}', encoding="utf-8")
    assert config_sync.load_template_batch_size(path, 12) == 7


def test_load_template_batch_size_uses_default_for_invalid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")
    assert config_sync.load_template_batch_size(path, 12) == 12


def test_persist_batch_size_rejects_values_below_one(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text('{"train": {"batch_size": 12}}', encoding="utf-8")
    monkeypatch.setattr(config_sync.gr, "update", lambda **kwargs: kwargs)
    update, message = config_sync.persist_batch_size(
        "demo",
        0,
        config_template_path=path,
        default_batch_size=12,
    )
    assert update == {"value": 12}
    assert "必须大于等于 1" in message
```

不依赖启动 Gradio 页面；对 `gr.update` 使用 monkeypatch。

- [ ] **Step 3: 为工作区识别编写测试**

覆盖空目录、直接 WAV、单说话人目录、多说话人目录、隐藏文件和非 WAV 文件。

- [ ] **Step 4: 为推理文件处理编写测试**

覆盖输出文件命名、允许的音频扩展名、重复文件名和不存在输入路径；测试仅调用纯路径逻辑。

- [ ] **Step 5: 运行测试并记录真实缺陷**

Run: `python -m pytest tests/test_path_utils.py tests/test_train_ui_config_sync.py tests/test_train_ui_workspace.py tests/test_infer_ui_files.py -v`

Expected: 现有行为与测试约定一致时 PASS。若暴露缺陷，只做最小修复，并在提交信息中明确说明；不得借机拆分模块。

- [ ] **Step 6: 运行完整轻量测试集**

Run: `python -m pytest tests -v`

Expected: PASS。

- [ ] **Step 7: 提交业务测试**

```bash
git add tests src
git commit -m "test: 覆盖发布关键路径与配置逻辑"
```

提交前确认 `git diff --cached --name-only` 中没有无关源码文件。

---

### Task 6: 更新 CI 为显式、只检查的发布门禁

**Files:**
- Replace: `.github/workflows/ruff.yml`
- Delete: `.github/workflows/reviewdog.yml`
- Create: `.github/workflows/release-check.yml`

**Interfaces:**
- Consumes: `requirements-dev.txt`、`scripts/check_release.py` 和 `tests/`。
- Produces: `lint`、`static-smoke`、`release-check` 三个独立状态。

- [ ] **Step 1: 重写 Ruff 工作流**

工作流必须：

```yaml
name: Lint
on: [push, pull_request]
jobs:
  ruff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install -r requirements-dev.txt
      - run: python -m ruff check src scripts tests
```

- [ ] **Step 2: 删除自动修复式工作流**

删除 `reviewdog.yml`，CI 不对 PR 自动生成代码修改。

- [ ] **Step 3: 添加发布检查工作流**

`release-check.yml` 包含两个 job：

```text
static-smoke
  compileall
  verify_app_smoke.py
  verify_stability_fixes.py

release-check
  安装 requirements-dev.txt 和 pyyaml
  pytest tests/test_release_checks.py tests/test_launchers.py -q
  python scripts/check_release.py
```

不安装完整音频和 PyTorch 依赖。`tests/test_path_utils.py`、`tests/test_train_ui_config_sync.py`、`tests/test_train_ui_workspace.py` 和 `tests/test_infer_ui_files.py` 会导入现有运行时模块，因此在安装完整项目依赖的发布验证环境中执行，不塞进轻量 CI。

- [ ] **Step 4: 本地执行 CI 等价命令**

Run: `python -m ruff check src scripts tests`

Expected: PASS；若发现历史源码问题，不全仓库自动修复，只对本轮新增文件修复，历史问题记录到发布限制。

Run: `python -m compileall -q src scripts tests`

Expected: PASS。

Run: `python -m pytest tests/test_release_checks.py tests/test_launchers.py -q`

Expected: PASS。

Run: `python scripts/check_release.py`

Expected: PASS。

- [ ] **Step 5: 提交 CI**

```bash
git add .github/workflows
git commit -m "ci: 添加发布稳定性检查"
```

---

### Task 7: 统一 Windows 启动脚本提示

**Files:**
- Modify: `launchers/启动训练界面.bat`
- Modify: `launchers/启动推理界面.bat`
- Modify: `launchers/启动tensorboard.bat`
- Create: `tests/test_launchers.py`

**Interfaces:**
- Preserves: `.venv311\Scripts\python.exe` 和现有 Python 模块入口。
- Produces: 一致的环境、退出码和日志提示。

- [ ] **Step 1: 编写启动脚本静态测试**

```python
from pathlib import Path


def test_launchers_keep_expected_python_entrypoints():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "启动训练界面.bat": "-m src.app_train",
        "启动推理界面.bat": "-m src.app_infer",
        "启动tensorboard.bat": "-m tensorboard.main",
    }
    for filename, command in expected.items():
        text = (root / "launchers" / filename).read_text(encoding="utf-8")
        assert '.venv311\\Scripts\\python.exe' in text
        assert command in text
        assert "exit /b %ERRORLEVEL%" in text
```

- [ ] **Step 2: 运行测试确认缺少退出码传播**

Run: `python -m pytest tests/test_launchers.py -v`

Expected: FAIL，缺少 `exit /b %ERRORLEVEL%`。

- [ ] **Step 3: 修改三个启动脚本**

每个脚本：

1. 保留 `chcp 65001`、`setlocal` 和项目根目录切换。
2. 虚拟环境存在后执行 `"%VENV_PYTHON%" --version`。
3. 缺少环境时指向 `docs\00_快速开始.md`。
4. 执行主命令后立即保存 `set "APP_EXIT_CODE=%ERRORLEVEL%"`。
5. 非零退出时显示退出码和日志/排错文档位置。
6. `pause` 后使用 `exit /b %APP_EXIT_CODE%` 返回真实状态。

- [ ] **Step 4: 验证脚本静态契约**

Run: `python -m pytest tests/test_launchers.py -v`

Expected: PASS。

在 Windows 人工验收时再验证双击行为；macOS/Linux 不伪造 Batch 执行结果。

- [ ] **Step 5: 提交启动脚本**

```bash
git add launchers tests/test_launchers.py
git commit -m "fix: 完善 Windows 启动错误提示"
```

---

### Task 8: 完整发布候选验证与记录

**Files:**
- Modify: `docs/90_开发与发布检查.md`
- Modify: `CHANGELOG.md`
- Modify only when validation exposes a scoped defect: affected file from Tasks 1-7

**Interfaces:**
- Consumes: 所有前序任务产物。
- Produces: 可重复的静态验证记录和 Windows 人工验收表。

- [ ] **Step 1: 检查工作区范围**

Run: `git status --short`

Expected: 只包含本任务预期的验证记录修改。

Run: `git diff --check`

Expected: 无空白错误。

- [ ] **Step 2: 运行完整静态验证**

Run: `python -m ruff check src scripts tests`

Expected: PASS，或只剩已记录且不由本轮引入的历史问题；发布前必须决定修复或明确阻断，不能静默忽略。

Run: `python -m compileall -q src scripts tests`

Expected: PASS。

Run: `python scripts/verify_app_smoke.py`

Expected: 输出两个 `[OK]`。

Run: `python scripts/verify_stability_fixes.py`

Expected: 输出四个 `[OK]` 和 `All stability-fix checks passed.`。

Run: `python -m pytest tests -q`

Expected: 全部 PASS。

Run: `python scripts/check_release.py`

Expected: `[OK] 发布前仓库检查通过。`

- [ ] **Step 3: 审查运行时行为未被修改**

Run: `git diff <design-commit>..HEAD -- config_templates src/train_pipeline src/models.py src/inference src/diffusion`

Expected: 除确由 Task 5 测试暴露并单独说明的最小修复外，不应存在运行时模型和训练逻辑改动。

- [ ] **Step 4: 填写 Windows 人工验收表**

在 `docs/90_开发与发布检查.md` 中保留可复制表格：

```text
系统版本
GPU 与显存
NVIDIA 驱动
Python
PyTorch
CUDA runtime
训练页启动
依赖状态检查
工作区创建
数据导入与预处理
主训练启动
推理页启动
模型加载
一次完整转换
日志和输出路径
```

没有真实 Windows/NVIDIA 验证结果时，表格状态写“未执行”，发布状态保持 Release Candidate，不填写虚假通过记录。

- [ ] **Step 5: 更新 Changelog 状态**

将已完成的文档、协作入口、检查、测试和启动提示列入“未发布”。不为版本号和发布日期做未经用户确认的假设。

- [ ] **Step 6: 提交验证记录**

```bash
git add docs/90_开发与发布检查.md CHANGELOG.md
git commit -m "docs: 完善发布候选验收记录"
```

- [ ] **Step 7: 输出发布候选报告**

报告必须包含：

- 已完成的发布改进。
- 所有实际运行的命令与结果。
- Windows 人工验收是否完成。
- 尚未验证的平台和功能。
- 当前分支与提交范围。
- 建议的版本号和发布说明草案，但不自动打 tag、不自动推送、不自动创建 GitHub Release。
