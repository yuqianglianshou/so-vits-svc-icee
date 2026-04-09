from __future__ import annotations
"""训练页中“模型工作区 / 训练语音数据”的展示辅助函数。

这里主要负责读取文件系统状态并整理成界面展示文本，不负责修改项目状态，
这样页面刷新逻辑会更稳定、也更容易复用。
"""

from datetime import datetime
from pathlib import Path

from src.train_ui.text import render_dataset_import_result


def _wav_files_in_dir(base_dir: Path):
    """返回目录下大小写不敏感识别到的 wav 文件。"""
    if not base_dir.exists():
        return []
    return sorted([path for path in base_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"])


def format_timestamp(ts) -> str:
    """把工作区里记录的时间戳格式化成页面展示文本。"""
    try:
        value = float(ts)
        if value <= 0:
            return "未记录"
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return "未记录"


def count_training_wavs(base_dir: Path):
    """统计处理后训练目录下的 wav 数量。"""
    if not base_dir.exists():
        return 0, 0
    direct_wavs = _wav_files_in_dir(base_dir)
    return (1 if direct_wavs else 0), len(direct_wavs)


def count_raw_dataset_wavs(base_dir: Path):
    """统计原始语音数据目录直接包含的 wav 数量。"""
    if not base_dir.exists() or not base_dir.is_dir():
        return 0, 0
    wavs = len(_wav_files_in_dir(base_dir))
    return (1 if wavs > 0 else 0), wavs


def has_raw_dataset_wavs(base_dir: Path):
    """判断当前原始语音数据目录里是否已有 wav。"""
    return count_raw_dataset_wavs(base_dir)[1] > 0


def speaker_dirs_in_train_root(base_dir: Path):
    """列出处理后训练目录下的一级子目录。"""
    if not base_dir.exists():
        return []
    if _wav_files_in_dir(base_dir):
        return [base_dir.name]
    return []


def raw_dataset_display_name(dataset_name: str):
    """为数据目录名提供一个不会为空的展示文本。"""
    return dataset_name or "未命名数据集"


def render_model_workspace_summary(root: Path, model_name: str, workspace, dataset_name: str, train_dir: str, raw_dataset_dir: Path):
    """渲染左侧“模型工作区”摘要卡片。"""
    if workspace is None:
        return (
            '<div class="dataset-import-result">'
            '<div class="dataset-import-result__title"><span class="stage-dot" style="color:#d97706;">●</span>模型工作区</div>'
            f'<div class="dataset-import-result__body" style="color:#d97706;">当前模型：{model_name}\n该模型还没有工作区信息，点击“新建训练模型”或切换已有模型。</div>'
            '</div>'
        )

    created_at = workspace.get("created_at", workspace.get("updated_at", 0))
    updated_at = workspace.get("updated_at", created_at)
    raw_dir_exists = (root / raw_dataset_dir).exists()
    raw_dir_line = "已存在" if raw_dir_exists else "未找到"
    created_at_text = format_timestamp(created_at)
    updated_at_text = format_timestamp(updated_at)
    return (
        '<div class="dataset-import-result">'
        '<div class="dataset-import-result__title"><span class="stage-dot" style="color:#1f8f4c;">●</span>模型工作区</div>'
        f'<div class="dataset-import-result__body" style="color:#1f8f4c;">当前模型：{model_name}；绑定模型数据目录：{dataset_name}；数据目录状态：{raw_dir_line}；处理目录：{train_dir}；创建时间：{created_at_text}；最近更新时间：{updated_at_text}</div>'
        '</div>'
    )


def render_dataset_file_list(root: Path, dataset_dir: Path):
    """渲染当前语音数据目录里的 wav 文件列表。"""
    if not dataset_dir.exists():
        return (
            '<div class="dataset-files-empty">'
            f'{dataset_dir.relative_to(root).as_posix()} 不存在。'
            '</div>'
        )
    wav_files = [path.name for path in _wav_files_in_dir(dataset_dir)]
    if not wav_files:
        return (
            '<div class="dataset-files-empty">'
            f'{dataset_dir.relative_to(root).as_posix()} 下暂无 wav 文件。'
            '</div>'
        )
    items = "".join(
        f'<div class="dataset-file-item">{name}</div>' for name in wav_files[:200]
    )
    more_line = ""
    if len(wav_files) > 200:
        more_line = f'<div class="dataset-files-more">其余 {len(wav_files) - 200} 个文件未展开</div>'
    return (
        '<div class="dataset-files-box">'
        f'<div class="dataset-files-head">当前目录：{dataset_dir.relative_to(root).as_posix()}</div>'
        f'<div class="dataset-files-count">文件数：{len(wav_files)}</div>'
        f'<div class="dataset-files-list">{items}</div>'
        f'{more_line}'
        '</div>'
    )


def dataset_file_list_label(wav_count: int):
    """生成“查看语音数据文件（*个）”这类折叠标题。"""
    return f"查看语音数据文件（{wav_count}个）"


def render_dataset_import_status_for_dataset(root: Path, dataset_dir: Path):
    """根据当前绑定的数据目录渲染“语音数据状态”卡片。"""
    _, wav_count = count_raw_dataset_wavs(dataset_dir)
    if wav_count > 0:
        return render_dataset_import_result(
            f"当前模型数据目录：{dataset_dir.relative_to(root).as_posix()}；已检测到 {wav_count} 个 wav。"
        )
    if dataset_dir.exists():
        return render_dataset_import_result(
            f"当前模型数据目录：{dataset_dir.relative_to(root).as_posix()}；目录已存在，但暂未检测到 wav。"
        )
    return render_dataset_import_result(
        f"当前模型数据目录：{dataset_dir.relative_to(root).as_posix()}；当前还没有导入 wav 数据。"
    )
