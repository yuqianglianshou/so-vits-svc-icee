from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr

IMPORTED_MODEL_ROOT = "./model_assets/imported_models"
WORKSPACE_MODEL_ROOT = "./model_assets/workspaces"


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (TypeError, ValueError):
        return -1


def _diffusion_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (TypeError, ValueError):
        return -1


def _list_trained_diffusion_checkpoints(diffusion_dir: Path):
    return [
        path
        for path in sorted(diffusion_dir.glob("model_*.pt"), key=_diffusion_step)
        if path.name != "model_0.pt"
    ]


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


def last_selected_infer_model_path(imported_model_root: str = IMPORTED_MODEL_ROOT) -> Path:
    return Path(imported_model_root) / "last_selected_infer_model.json"


def save_last_selected_infer_model(model_source: str, model_dir: str, imported_model_root: str = IMPORTED_MODEL_ROOT):
    safe_source = (model_source or "").strip()
    safe_value = (model_dir or "").strip()
    payload = {
        "model_source": safe_source,
        "model_dir": safe_value,
        "updated_at": int(time.time()),
    }
    target = last_selected_infer_model_path(imported_model_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return safe_source, safe_value


def load_last_selected_infer_model(imported_model_root: str = IMPORTED_MODEL_ROOT):
    path = last_selected_infer_model_path(imported_model_root)
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    model_source = str(payload.get("model_source") or "").strip() or None
    model_dir = str(payload.get("model_dir") or "").strip() or None
    if model_dir and model_dir.startswith("./model_assets/local/"):
        model_dir = model_dir.replace("./model_assets/local/", f"{IMPORTED_MODEL_ROOT}/", 1)
        model_source = model_source or "imported"
    elif model_dir and model_dir.startswith("model_assets/local/"):
        model_dir = model_dir.replace("model_assets/local/", f"{IMPORTED_MODEL_ROOT}/", 1)
        model_source = model_source or "imported"
    return model_source, model_dir


def scan_imported_models(imported_model_root: str = IMPORTED_MODEL_ROOT):
    return _find_loadable_model_dirs(imported_model_root)


def scan_workspace_models(workspace_model_root: str = WORKSPACE_MODEL_ROOT):
    return _find_loadable_model_dirs(workspace_model_root)


def list_local_model_checkpoints(local_model_selection: str):
    model_dir = Path((local_model_selection or "").strip())
    if not model_dir.exists():
        return []
    return [str(path) for path in sorted(model_dir.glob("G_*.pth"), key=_checkpoint_step)]


def list_local_model_diffusion_checkpoints(local_model_selection: str):
    model_dir = Path((local_model_selection or "").strip())
    diffusion_dir = model_dir / "diffusion"
    if not diffusion_dir.is_dir():
        return []
    return [str(path) for path in _list_trained_diffusion_checkpoints(diffusion_dir)]


def detect_local_model_extras(local_model_selection: str):
    model_dir = Path((local_model_selection or "").strip())
    if not model_dir.exists():
        return "", "", ""

    diff_model_path = ""
    diff_config_path = ""
    cluster_model_path = ""
    diffusion_dir = model_dir / "diffusion"
    if diffusion_dir.is_dir():
        candidates = _list_trained_diffusion_checkpoints(diffusion_dir)
        if candidates:
            diff_model_path = str(candidates[-1])

    diff_config_candidate = model_dir / "diffusion.yaml"
    if diff_config_candidate.is_file():
        diff_config_path = str(diff_config_candidate)

    cluster_candidate = model_dir / "feature_and_index.pkl"
    if cluster_candidate.is_file():
        cluster_model_path = str(cluster_candidate)

    return diff_model_path, diff_config_path, cluster_model_path


def model_checkpoint_refresh_fn(model_selection: str):
    checkpoints = list_local_model_checkpoints(model_selection)
    value = checkpoints[-1] if checkpoints else None
    return gr.update(choices=checkpoints, value=value, interactive=bool(checkpoints))


def model_diffusion_checkpoint_refresh_fn(model_selection: str):
    checkpoints = list_local_model_diffusion_checkpoints(model_selection)
    value = checkpoints[-1] if checkpoints else None
    return gr.update(choices=checkpoints, value=value, interactive=bool(checkpoints))


def model_extra_refresh_fn(model_selection: str):
    diff_model_path, diff_config_path, cluster_model_path = detect_local_model_extras(model_selection)
    return (
        gr.update(value=diff_model_path or "", choices=list_local_model_diffusion_checkpoints(model_selection), interactive=bool(diff_model_path)),
        gr.update(value=diff_config_path or ""),
        gr.update(value=cluster_model_path or ""),
    )


def model_option_refresh_fn(model_selection: str):
    diff_model_path, diff_config_path, cluster_model_path = detect_local_model_extras(model_selection)
    has_diffusion = bool(diff_model_path and diff_config_path)
    has_cluster = bool(cluster_model_path)
    return (
        gr.update(value=has_diffusion, interactive=has_diffusion),
        gr.update(value=has_cluster, interactive=has_cluster),
    )


def imported_model_refresh_fn(imported_model_root: str = IMPORTED_MODEL_ROOT):
    choices = scan_imported_models(imported_model_root)
    remembered_source, remembered_dir = load_last_selected_infer_model(imported_model_root)
    value = remembered_dir if remembered_source == "imported" and remembered_dir in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def workspace_model_refresh_fn(workspace_model_root: str = WORKSPACE_MODEL_ROOT, imported_model_root: str = IMPORTED_MODEL_ROOT):
    choices = scan_workspace_models(workspace_model_root)
    remembered_source, remembered_dir = load_last_selected_infer_model(imported_model_root)
    value = remembered_dir if remembered_source == "workspace" and remembered_dir in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def persist_selected_model(model_source, model_selection, imported_model_root: str = IMPORTED_MODEL_ROOT):
    if model_selection:
        save_last_selected_infer_model(model_source, model_selection, imported_model_root)
    return gr.update()
