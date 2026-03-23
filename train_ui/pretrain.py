from __future__ import annotations
"""“训练前依赖与底模”区域的 UI 辅助函数。

这里把依赖项解析、状态判断和展示逻辑从主训练页里拆出来，让 app_train.py
能更专注在流程编排和事件绑定上。
"""

import ast
from pathlib import Path


def resolve_uploaded_path(file_obj):
    """把 Gradio 文件值统一转换成可用的本地路径字符串。"""
    if file_obj is None:
        return ""
    return getattr(file_obj, "name", file_obj)


def normalize_asset_key(asset_key, asset_registry):
    """把下拉框返回值还原成内部统一使用的依赖 key。"""
    if asset_key in asset_registry:
        return asset_key

    if isinstance(asset_key, str):
        for key, asset in asset_registry.items():
            if asset_key == asset["label"]:
                return key
        try:
            parsed = ast.literal_eval(asset_key)
            if isinstance(parsed, tuple) and len(parsed) >= 2 and parsed[1] in asset_registry:
                return parsed[1]
        except (ValueError, SyntaxError):
            pass

    raise KeyError(asset_key)


def render_pretrain_target(root: Path, asset):
    """返回界面里展示用的项目相对目标路径。"""
    if asset.get("is_archive"):
        return asset["target"].relative_to(root).as_posix() + "/"
    return asset["target"].relative_to(root).as_posix()


def is_pretrain_asset_ready(asset, rmvpe_path: Path, rmvpe_validator):
    """判断某个训练前依赖是否已经就绪且可用。"""
    if asset.get("is_archive"):
        target_dir = asset["target"]
        required_files = asset.get("required_files")
        if required_files:
            return all((target_dir / relative_path).exists() for relative_path in required_files)
        return (target_dir / "model").exists() and (target_dir / "config.json").exists()
    if asset["target"] == rmvpe_path:
        return rmvpe_validator()
    return asset["target"].exists()


def pretrain_asset_state(asset_key: str, asset_registry, rmvpe_path: Path, rmvpe_validator):
    """返回单个依赖的简短状态文案。"""
    asset = asset_registry[asset_key]
    if asset.get("is_archive"):
        return "已存在" if is_pretrain_asset_ready(asset, rmvpe_path, rmvpe_validator) else "缺失"
    if asset_key == "rmvpe":
        if not asset["target"].exists():
            return "缺失"
        if not rmvpe_validator():
            return "损坏"
        return "已存在"
    return "已存在" if asset["target"].exists() else "缺失"


def first_missing_pretrain_asset(asset_registry, rmvpe_path: Path, rmvpe_validator):
    """找出当前最优先需要补齐的依赖。"""
    for key, asset in asset_registry.items():
        if not is_pretrain_asset_ready(asset, rmvpe_path, rmvpe_validator):
            return key
    return None


def ordered_pretrain_asset_keys(asset_registry, rmvpe_path: Path, rmvpe_validator):
    """把依赖项排序成“缺失在前、已就绪在后”。"""
    missing = []
    ready = []
    for key, asset in asset_registry.items():
        if is_pretrain_asset_ready(asset, rmvpe_path, rmvpe_validator):
            ready.append(key)
        else:
            missing.append(key)
    return missing + ready


def pretrain_asset_choices(asset_registry, rmvpe_path: Path, rmvpe_validator):
    """生成带状态前缀的依赖下拉选项。"""
    choices = []
    for key in ordered_pretrain_asset_keys(asset_registry, rmvpe_path, rmvpe_validator):
        prefix = "✓ " if is_pretrain_asset_ready(asset_registry[key], rmvpe_path, rmvpe_validator) else "✕ "
        choices.append((prefix + asset_registry[key]["label"], key))
    return choices


def next_missing_pretrain_asset(current_key: str, asset_registry, rmvpe_path: Path, rmvpe_validator):
    """从当前依赖开始，找到后面下一个仍缺失的依赖。"""
    keys = list(asset_registry.keys())
    if current_key not in asset_registry:
        current_key = keys[0]
    start_index = keys.index(current_key)
    for offset in range(1, len(keys) + 1):
        key = keys[(start_index + offset) % len(keys)]
        if not is_pretrain_asset_ready(asset_registry[key], rmvpe_path, rmvpe_validator):
            return key
    return current_key


def render_pretrain_status(root: Path, asset_registry, rmvpe_path: Path, rmvpe_validator):
    """渲染训练前依赖与底模的状态清单卡片。"""
    rows = []
    for asset_key, asset in asset_registry.items():
        status = pretrain_asset_state(asset_key, asset_registry, rmvpe_path, rmvpe_validator)
        ready = status == "已存在"
        status_icon = "✓" if ready else "✕"
        color = "#1f8f4c" if ready else "#c0392b"
        rows.append(
            '<div class="dependency-status-row">'
            f'<div class="dependency-status-title"><span class="dependency-status-icon" style="color:{color};border-color:{color};">{status_icon}</span>{asset["label"]}：<span style="color:{color};">{status}</span></div>'
            f'<div class="dependency-status-path">{render_pretrain_target(root, asset)}</div>'
            '</div>'
        )

    all_ready = all(is_pretrain_asset_ready(asset, rmvpe_path, rmvpe_validator) for asset in asset_registry.values())
    summary = "训练前依赖与底模已就绪。" if all_ready else "训练前依赖与底模还不完整。"
    summary_color = "#1f8f4c" if all_ready else "#c0392b"
    rows.append(f'<div class="dependency-status-summary" style="color:{summary_color};">{summary}</div>')
    return '<div class="stage-progress-box">' + "".join(rows) + '</div>'


def render_pretrain_asset_guide(root: Path, asset_key, asset_registry, rmvpe_path: Path, rmvpe_validator):
    """渲染当前选中依赖的说明、目标位置和下载提示。"""
    asset_key = normalize_asset_key(asset_key, asset_registry)
    asset = asset_registry[asset_key]
    status = pretrain_asset_state(asset_key, asset_registry, rmvpe_path, rmvpe_validator)
    links = " / ".join([f'<a href="{url}" target="_blank">{label}</a>' for label, url in asset["download_links"]])
    return (
        '<div style="font-size:14px; color:#5f5a52;">'
        f"<div><strong>{asset['label']}</strong> · {status}</div>"
        f"<div>作用：{asset['purpose']}</div>"
        f"<div>目标位置：{render_pretrain_target(root, asset)}</div>"
        f"<div>下载：{links}</div>"
        "</div>"
    )
