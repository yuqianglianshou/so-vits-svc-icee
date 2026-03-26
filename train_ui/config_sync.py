from __future__ import annotations

import json

import gradio as gr

from train_ui.paths import model_config_path, sanitize_model_name


def load_template_batch_size(config_template_path, default_batch_size: int) -> int:
    try:
        payload = json.loads(config_template_path.read_text(encoding="utf-8"))
        return int(payload.get("train", {}).get("batch_size", default_batch_size))
    except Exception:
        return default_batch_size


def load_model_batch_size(model_name: str, config_template_path, default_batch_size: int) -> int:
    model_name = sanitize_model_name(model_name) or "default_model"
    config_path = model_config_path(model_name)
    if not config_path.exists():
        return load_template_batch_size(config_template_path, default_batch_size)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return int(config.get("train", {}).get("batch_size", load_template_batch_size(config_template_path, default_batch_size)))
    except Exception:
        return load_template_batch_size(config_template_path, default_batch_size)


def persist_batch_size(
    model_name: str,
    batch_size: int | float | None,
    *,
    config_template_path,
    config_tiny_template_path,
    default_batch_size: int,
):
    if batch_size in {None, ""}:
        current_value = load_model_batch_size(model_name, config_template_path, default_batch_size)
        return gr.update(value=current_value), "batch size 不能为空，已恢复为当前配置值。"

    try:
        batch_size = int(batch_size)
    except Exception:
        current_value = load_model_batch_size(model_name, config_template_path, default_batch_size)
        return gr.update(value=current_value), "batch size 必须是正整数，已恢复为当前配置值。"

    if batch_size < 1:
        current_value = load_model_batch_size(model_name, config_template_path, default_batch_size)
        return gr.update(value=current_value), "batch size 必须大于等于 1，已恢复为当前配置值。"

    template_paths = [config_template_path, config_tiny_template_path]
    for template_path in template_paths:
        if template_path.exists():
            payload = json.loads(template_path.read_text(encoding="utf-8"))
            payload.setdefault("train", {})["batch_size"] = batch_size
            template_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    safe_model_name = sanitize_model_name(model_name) or "default_model"
    config_path = model_config_path(safe_model_name)
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(config_template_path.read_text(encoding="utf-8"))
    payload.setdefault("train", {})["batch_size"] = batch_size
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return (
        gr.update(value=batch_size),
        f"已将 batch size 保存为 {batch_size}，并同步到当前模型配置与模板配置。",
    )
