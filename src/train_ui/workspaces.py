from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import gradio as gr

from src.train_ui.paths import (
    ROOT,
    default_train_dir_for_dataset,
    model_root_dir,
    model_workspace_path,
    resolve_raw_dataset_dir,
    sanitize_dataset_name,
    sanitize_model_name,
)
from src.train_ui.pretrain import resolve_uploaded_path
from src.train_ui.text import render_dataset_import_result
from src.train_ui.workspace import (
    has_raw_dataset_wavs,
    raw_dataset_display_name as build_raw_dataset_display_name,
    render_dataset_file_list as render_dataset_file_list_html,
    render_dataset_import_status_for_dataset as render_dataset_import_status_for_dataset_html,
    render_model_workspace_summary as render_model_workspace_summary_html,
)


RAW_DATASET_PARENT = ROOT / "training_data/source"


def ensure_raw_dataset_parent():
    RAW_DATASET_PARENT.mkdir(parents=True, exist_ok=True)


def raw_dataset_display_name(dataset_name: str):
    return build_raw_dataset_display_name(sanitize_dataset_name(dataset_name) or "")


def ensure_model_workspace_dirs(model_name: str):
    root = model_root_dir(model_name)
    (root / "diffusion").mkdir(parents=True, exist_ok=True)
    (root / "filelists").mkdir(parents=True, exist_ok=True)
    return root


def save_model_workspace(model_name: str, dataset_name: str):
    model_name = sanitize_model_name(model_name)
    dataset_name = model_name
    train_dir = default_train_dir_for_dataset(dataset_name)
    ensure_model_workspace_dirs(model_name)
    payload = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "train_dir": train_dir,
        "updated_at": int(time.time()),
    }
    workspace_path = model_workspace_path(model_name)
    if workspace_path.exists():
        try:
            previous = json.loads(workspace_path.read_text(encoding="utf-8"))
            if "created_at" in previous:
                payload["created_at"] = previous["created_at"]
        except Exception:
            pass
    payload.setdefault("created_at", payload["updated_at"])
    workspace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_model_workspace(model_name: str):
    workspace_path = model_workspace_path(model_name)
    if not workspace_path.exists():
        return None
    try:
        return json.loads(workspace_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_workspace_dataset_name(model_name: str):
    return sanitize_model_name(model_name)


def is_model_workspace_candidate(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = (
        path / "workspace.json",
        path / "config.json",
        path / "diffusion.yaml",
        path / "feature_and_index.pkl",
        path / "diffusion",
        path / "filelists",
    )
    if any(marker.exists() for marker in markers):
        return True
    return any(path.glob("G_*.pth")) or any(path.glob("D_*.pth"))


def scan_model_workspaces():
    logs_dir = ROOT / "model_assets/workspaces"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ensure_raw_dataset_parent()
    models = set()
    for child in sorted(logs_dir.iterdir()):
        if child.name == "webui_tasks" or not is_model_workspace_candidate(child):
            continue
        workspace = load_model_workspace(child.name)
        expected_dataset_name = infer_workspace_dataset_name(child.name)
        if workspace is None or workspace.get("dataset_name") != expected_dataset_name:
            workspace = save_model_workspace(child.name, expected_dataset_name)
        models.add(workspace["model_name"])
    for child in sorted(RAW_DATASET_PARENT.iterdir()):
        if child.is_dir() and child.name != "webui_tasks":
            models.add(infer_workspace_dataset_name(child.name))
    models = sorted(models)
    if not models:
        models = ["default_model"]
    return models


def last_selected_model_path() -> Path:
    return ROOT / "model_assets/workspaces" / "last_selected_model.json"


def save_last_selected_model(model_name: str):
    safe_name = sanitize_model_name(model_name)
    payload = {
        "model_name": safe_name,
        "updated_at": int(time.time()),
    }
    last_selected_model_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return safe_name


def load_last_selected_model():
    path = last_selected_model_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model_name = sanitize_model_name(payload.get("model_name"))
    return model_name or None


def resolve_model_choice(model_name: str):
    safe_name = sanitize_model_name(model_name)
    candidates = scan_model_workspaces()
    if safe_name in candidates:
        return safe_name
    return candidates[0] if candidates else "default_model"


def render_model_workspace_summary(model_name: str):
    model_name = sanitize_model_name(model_name)
    workspace = load_model_workspace(model_name)
    dataset_name = model_name
    train_dir = workspace.get("train_dir", default_train_dir_for_dataset(dataset_name)) if workspace else default_train_dir_for_dataset(dataset_name)
    return render_model_workspace_summary_html(
        ROOT,
        model_name,
        workspace,
        dataset_name,
        train_dir,
        ROOT / resolve_raw_dataset_dir(dataset_name),
    )


def scan_dataset_candidates():
    ensure_raw_dataset_parent()
    candidates = []
    for child in sorted(RAW_DATASET_PARENT.iterdir()):
        if child.is_dir() and has_raw_dataset_wavs(child):
            candidates.append(child.name)
    if not candidates:
        candidates.append("default_dataset")
    return candidates


def suggest_next_dataset_name():
    ensure_raw_dataset_parent()
    index = 1
    while True:
        candidate = f"speak {index}"
        if not (RAW_DATASET_PARENT / candidate).exists():
            return candidate
        index += 1


def infer_uploaded_dataset_name(uploaded_files):
    if not uploaded_files:
        return None
    file_items = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    parent_names = []
    ignored_names = {"", ".", "..", "tmp", "temp", "gradio"}
    for item in file_items:
        original_name = getattr(item, "orig_name", "") or ""
        parent_name = Path(original_name).parent.name if original_name else ""
        if parent_name.lower() in ignored_names:
            continue
        if len(parent_name) >= 24 and all(ch in "0123456789abcdef" for ch in parent_name.lower()):
            continue
        parent_names.append(parent_name)
    if not parent_names:
        return None
    unique_names = {name for name in parent_names if name}
    if len(unique_names) == 1:
        return unique_names.pop()
    return None


def import_dataset_directory(uploaded_files, dataset_name: str):
    dataset_name = sanitize_dataset_name(dataset_name) or "default_dataset"
    if not uploaded_files:
        return render_dataset_import_result("请先选择一个数据集文件夹。"), gr.update(value=dataset_name)

    target_relative_dir = resolve_raw_dataset_dir(dataset_name)
    target_root = ROOT / target_relative_dir
    target_root.mkdir(parents=True, exist_ok=True)

    file_items = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    copied = 0
    overwritten = 0
    for item in file_items:
        source_path = Path(resolve_uploaded_path(item))
        original_name = getattr(item, "orig_name", "") or ""
        file_name = Path(original_name).name if original_name else source_path.name
        if not file_name.lower().endswith(".wav"):
            continue
        destination = target_root / file_name
        if destination.exists():
            overwritten += 1
        shutil.copyfile(source_path, destination)
        copied += 1

    if copied == 0:
        return render_dataset_import_result("导入失败：没有检测到 wav 文件。请选择一个内部直接包含 `.wav` 文件的模型数据文件夹。"), gr.update(value=dataset_name)

    return (
        render_dataset_import_result(
            f"本次已导入到：{target_relative_dir.as_posix()}；当前模型数据目录：{raw_dataset_display_name(dataset_name)}；导入音频数：{copied}；覆盖同名文件数：{overwritten}"
        ),
        gr.update(value=dataset_name),
    )


def render_dataset_file_list(dataset_name: str):
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    return render_dataset_file_list_html(ROOT, dataset_dir)


def dataset_file_list_label(dataset_name: str):
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    wav_count = len([
        path for path in dataset_dir.iterdir()
        if dataset_dir.exists() and path.is_file() and path.suffix.lower() == ".wav"
    ]) if dataset_dir.exists() else 0
    return f"查看语音数据文件（{wav_count}个）"


def render_dataset_import_status_for_dataset(dataset_name: str):
    dataset_name = sanitize_dataset_name(dataset_name) or "default_dataset"
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    return render_dataset_import_status_for_dataset_html(ROOT, dataset_dir)


def delete_dataset_directory(dataset_name: str):
    dataset_name = sanitize_dataset_name(dataset_name) or "default_dataset"
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    current_train_dir = default_train_dir_for_dataset(dataset_name)
    if not dataset_dir.exists():
        return (
            render_dataset_import_result(f"{dataset_dir.relative_to(ROOT).as_posix()} 不存在，无需删除。"),
            gr.update(value=dataset_name),
            render_dataset_file_list(dataset_name),
            gr.update(value=dataset_name),
            gr.update(value=current_train_dir),
            gr.update(value=dataset_name),
        )

    if dataset_dir == RAW_DATASET_PARENT:
        return (
            render_dataset_import_result("禁止删除 training_data/source 父目录。"),
            gr.update(value=dataset_name),
            render_dataset_file_list(dataset_name),
            gr.update(value=dataset_name),
            gr.update(value=current_train_dir),
            gr.update(value=dataset_name),
        )

    shutil.rmtree(dataset_dir)
    return (
        render_dataset_import_result(f"已删除模型数据目录：{dataset_dir.relative_to(ROOT).as_posix()}"),
        gr.update(value=dataset_name),
        render_dataset_file_list(dataset_name),
        gr.update(value=dataset_name),
        gr.update(value=current_train_dir),
        gr.update(value=dataset_name),
    )


def prepare_delete_dataset(dataset_name: str):
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    if not dataset_dir.exists():
        return (
            f"**{dataset_dir.relative_to(ROOT).as_posix()}** 不存在，无需删除。",
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(),
            gr.update(visible=False),
        )
    wav_count = len([path for path in dataset_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"])
    message = (
        f"确认删除：{dataset_dir.relative_to(ROOT).as_posix()}\n\n"
        f"该目录下共有 {wav_count} 个 wav 文件。\n"
        "这个操作不会删除 model_assets/workspaces 下的模型工作区。"
    )
    return (
        message,
        gr.update(visible=True),
        gr.update(value=dataset_name),
        gr.update(value="dataset"),
        gr.update(),
        gr.update(visible=False),
    )


def create_model_workspace_action(
    new_model_name: str,
    current_dataset_name: str,
    *,
    has_active_task_fn,
    active_task_block_message_fn,
    speech_encoder_value_update_fn,
    load_model_speech_encoder_fn,
):
    if has_active_task_fn():
        current_model = sanitize_model_name(current_dataset_name) or "default_model"
        current_dataset = sanitize_dataset_name(current_dataset_name) or current_model
        return (
            gr.update(value=current_model),
            gr.update(value=current_model),
            gr.update(value=current_model),
            gr.update(value=current_dataset),
            gr.update(value=default_train_dir_for_dataset(current_dataset)),
            gr.update(value=sanitize_model_name(current_dataset) or "default_model"),
            speech_encoder_value_update_fn(load_model_speech_encoder_fn(current_model)),
            render_model_workspace_summary(current_model),
            render_dataset_import_result(active_task_block_message_fn("新建模型工作区")),
        )
    model_name = sanitize_model_name(new_model_name)
    dataset_name = model_name
    workspace_exists = model_workspace_path(model_name).exists()
    save_model_workspace(model_name, dataset_name)
    save_last_selected_model(model_name)
    return (
        gr.update(choices=scan_model_workspaces(), value=model_name),
        gr.update(value=model_name),
        gr.update(value=model_name),
        gr.update(value=dataset_name),
        gr.update(value=default_train_dir_for_dataset(dataset_name)),
        gr.update(value=sanitize_model_name(dataset_name) or "default_model"),
        speech_encoder_value_update_fn(load_model_speech_encoder_fn(model_name)),
        render_model_workspace_summary(model_name),
        render_dataset_import_result(
            f"{'已切换到已有模型工作区' if workspace_exists else '已新建模型工作区'}：{model_name}；绑定模型数据目录：{dataset_name}"
        ),
    )


def switch_model_workspace_action(
    selected_model_name: str,
    current_model_name: str,
    current_dataset_name: str,
    current_train_dir: str,
    current_sync_token: str,
    *,
    active_task_name: str | None,
    has_active_task_fn,
    speech_encoder_value_update_fn,
    load_model_speech_encoder_fn,
    active_task_block_message_fn,
    show_active_task_block_dialog_fn,
):
    if has_active_task_fn():
        return (
            gr.update(value=sanitize_model_name(current_model_name) or "default_model"),
            gr.update(value=sanitize_model_name(current_model_name) or "default_model"),
            gr.update(value=sanitize_model_name(current_model_name) or "default_model"),
            gr.update(value=sanitize_dataset_name(current_dataset_name) or "default_dataset"),
            gr.update(value=current_train_dir or default_train_dir_for_dataset(current_dataset_name)),
            gr.update(value=current_sync_token),
            speech_encoder_value_update_fn(load_model_speech_encoder_fn(current_model_name)),
            render_model_workspace_summary(sanitize_model_name(current_model_name) or "default_model"),
            render_dataset_import_result(active_task_block_message_fn("切换模型")),
            *show_active_task_block_dialog_fn("切换模型"),
        )
    model_name = resolve_model_choice(selected_model_name)
    dataset_name = infer_workspace_dataset_name(model_name)
    save_model_workspace(model_name, dataset_name)
    save_last_selected_model(model_name)
    raw_dir_exists = (ROOT / resolve_raw_dataset_dir(dataset_name)).exists()
    return (
        gr.update(choices=scan_model_workspaces(), value=model_name),
        gr.update(value=model_name),
        gr.update(value=model_name),
        gr.update(value=dataset_name),
        gr.update(value=default_train_dir_for_dataset(dataset_name)),
        gr.update(value=sanitize_model_name(dataset_name) or "default_model"),
        speech_encoder_value_update_fn(load_model_speech_encoder_fn(model_name)),
        render_model_workspace_summary(model_name),
        render_dataset_import_result(
            f"已切换到模型工作区：{model_name}；绑定模型数据目录：{dataset_name}；数据目录：{'已存在' if raw_dir_exists else '未找到，请导入 wav 数据'}"
        ),
        gr.update(),
        gr.update(visible=False),
    )


def prepare_delete_model_workspace_action(
    selected_model_name: str,
    *,
    has_active_task_fn,
    active_task_block_message_fn,
    show_active_task_block_dialog_fn,
):
    if has_active_task_fn():
        return (
            "",
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            render_dataset_import_result(active_task_block_message_fn("删除模型工作区")),
            *show_active_task_block_dialog_fn("删除模型工作区"),
        )
    model_name = resolve_model_choice(selected_model_name)
    model_dir = model_root_dir(model_name)
    dataset_name = infer_workspace_dataset_name(model_name)
    raw_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    train_dir = ROOT / default_train_dir_for_dataset(dataset_name)
    if not model_dir.exists() and not raw_dir.exists() and not train_dir.exists():
        return (
            f"**model_assets/workspaces/{model_name}**、**{resolve_raw_dataset_dir(dataset_name).as_posix()}**、**{default_train_dir_for_dataset(dataset_name)}** 都不存在，无需删除。",
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
        )
    message = (
        f"确认删除：{model_dir.relative_to(ROOT).as_posix()}\n\n"
        f"当前模型：{model_name}\n"
        f"绑定模型数据目录：{dataset_name}\n\n"
        "这个操作会一并删除以下内容：\n"
        f"- model_assets/workspaces/{model_name}\n"
        f"- {resolve_raw_dataset_dir(dataset_name).as_posix()}\n"
        f"- {default_train_dir_for_dataset(dataset_name)}"
    )
    return (
        message,
        gr.update(visible=True),
        gr.update(value=model_name),
        gr.update(value="workspace"),
        gr.update(),
        gr.update(),
        gr.update(visible=False),
    )


def delete_model_workspace_action(
    selected_model_name: str,
    *,
    speech_encoder_value_update_fn,
    load_model_speech_encoder_fn,
):
    model_name = resolve_model_choice(selected_model_name)
    model_dir = model_root_dir(model_name)
    dataset_name = infer_workspace_dataset_name(model_name)
    raw_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    train_dir = ROOT / default_train_dir_for_dataset(dataset_name)
    deleted_targets = []

    for target in (model_dir, raw_dir, train_dir):
        if target.exists():
            shutil.rmtree(target)
            deleted_targets.append(target.relative_to(ROOT).as_posix())

    if not deleted_targets:
        next_model = resolve_model_choice("default_model")
        next_dataset = infer_workspace_dataset_name(next_model)
        save_last_selected_model(next_model)
        return (
            gr.update(choices=scan_model_workspaces(), value=next_model),
            gr.update(value=next_model),
            gr.update(value=next_model),
            gr.update(value=next_dataset),
            gr.update(value=default_train_dir_for_dataset(next_dataset)),
            gr.update(value=sanitize_model_name(next_dataset) or "default_model"),
            speech_encoder_value_update_fn(load_model_speech_encoder_fn(next_model)),
            render_model_workspace_summary(next_model),
            render_dataset_import_result(
                f"model_assets/workspaces/{model_name}、{resolve_raw_dataset_dir(dataset_name).as_posix()}、{default_train_dir_for_dataset(dataset_name)} 都不存在，无需删除。"
            ),
        )
    remaining_models = scan_model_workspaces()
    next_model = remaining_models[0] if remaining_models else "default_model"
    next_dataset = infer_workspace_dataset_name(next_model)
    save_model_workspace(next_model, next_dataset)
    save_last_selected_model(next_model)
    return (
        gr.update(choices=remaining_models, value=next_model),
        gr.update(value=next_model),
        gr.update(value=next_model),
        gr.update(value=next_dataset),
        gr.update(value=default_train_dir_for_dataset(next_dataset)),
        gr.update(value=sanitize_model_name(next_dataset) or "default_model"),
        speech_encoder_value_update_fn(load_model_speech_encoder_fn(next_model)),
        render_model_workspace_summary(next_model),
        render_dataset_import_result(
            f"已删除模型相关目录：{'；'.join(deleted_targets)}；当前已切换到：{next_model}"
        ),
    )
