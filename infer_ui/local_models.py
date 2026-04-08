from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr

LOCAL_MODEL_ROOT = "./model_assets/local"
FALLBACK_MODEL_ROOT = "./model_assets/workspaces"


def _find_loadable_model_dirs(root_dir: str):
    root_path = Path(root_dir)
    if not root_path.exists():
        return []

    candidates = []
    for config_path in root_path.rglob("config.json"):
        model_dir = config_path.parent
        if list(model_dir.glob("G_*.pth")):
            candidates.append(str(model_dir))
    return sorted(set(candidates))


def last_selected_infer_model_path(local_model_root: str = LOCAL_MODEL_ROOT) -> Path:
    return Path(local_model_root) / "last_selected_infer_model.json"


def save_last_selected_infer_model(model_dir: str, local_model_root: str = LOCAL_MODEL_ROOT):
    safe_value = (model_dir or "").strip()
    payload = {
        "model_dir": safe_value,
        "updated_at": int(time.time()),
    }
    target = last_selected_infer_model_path(local_model_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return safe_value


def load_last_selected_infer_model(local_model_root: str = LOCAL_MODEL_ROOT):
    path = last_selected_infer_model_path(local_model_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model_dir = str(payload.get("model_dir") or "").strip()
    return model_dir or None


def scan_local_models(local_model_root: str = LOCAL_MODEL_ROOT):
    local_dirs = _find_loadable_model_dirs(local_model_root)
    fallback_dirs = _find_loadable_model_dirs(FALLBACK_MODEL_ROOT)
    return sorted(set(local_dirs + fallback_dirs))


def local_model_refresh_fn(local_model_root: str = LOCAL_MODEL_ROOT):
    choices = scan_local_models(local_model_root)
    remembered = load_last_selected_infer_model(local_model_root)
    value = remembered if remembered in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def persist_local_model_selection(local_model_selection, local_model_root: str = LOCAL_MODEL_ROOT):
    if local_model_selection:
        save_last_selected_infer_model(local_model_selection, local_model_root)
    return gr.update()
