from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from src.infer_ui.local_models import LOCAL_MODEL_ROOT
from src.train_ui.paths import sanitize_model_name


def resolve_uploaded_path(file_obj):
    if file_obj is None:
        return ""
    return getattr(file_obj, "name", file_obj)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (TypeError, ValueError):
        return -1


def _uploaded_path_obj(file_obj) -> Path | None:
    resolved_path = resolve_uploaded_path(file_obj)
    if not resolved_path:
        return None
    return Path(resolved_path)


def _uploaded_display_name(file_obj) -> str:
    original_name = getattr(file_obj, "orig_name", None)
    if original_name:
        return Path(original_name).name
    resolved_path = resolve_uploaded_path(file_obj)
    return Path(resolved_path).name if resolved_path else ""


def _safe_copy(source: Path | None, target: Path | None) -> str:
    if source is None or target is None:
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def _infer_local_model_dir_name(model_file, config_file) -> str:
    config_path = _uploaded_path_obj(config_file)
    if config_path and config_path.exists():
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            speakers = list((config_payload.get("spk") or {}).keys())
            if len(speakers) == 1:
                inferred_name = sanitize_model_name(speakers[0])
                if inferred_name and inferred_name != "default_model":
                    return inferred_name
        except Exception:
            pass

    model_name = Path(_uploaded_display_name(model_file)).stem
    model_name = re.sub(r"^G_\d+$", "uploaded_model", model_name)
    inferred_name = sanitize_model_name(model_name)
    return inferred_name or "uploaded_model"


def _store_uploaded_model_bundle(model_path, config_path, cluster_model_path, diff_model_path, diff_config_path):
    model_source = _uploaded_path_obj(model_path)
    config_source = _uploaded_path_obj(config_path)
    if model_source is None or config_source is None:
        raise FileNotFoundError("手动上传模式至少需要主模型 `.pth` 和配置 `.json`。")

    local_model_dir = Path(LOCAL_MODEL_ROOT) / _infer_local_model_dir_name(model_path, config_path)
    local_model_dir.mkdir(parents=True, exist_ok=True)

    stored_model_path = _safe_copy(model_source, local_model_dir / _uploaded_display_name(model_path))
    stored_config_path = _safe_copy(config_source, local_model_dir / "config.json")
    stored_cluster_path = _safe_copy(
        _uploaded_path_obj(cluster_model_path),
        local_model_dir / _uploaded_display_name(cluster_model_path) if cluster_model_path else None,
    )
    stored_diff_model_path = _safe_copy(
        _uploaded_path_obj(diff_model_path),
        local_model_dir / _uploaded_display_name(diff_model_path) if diff_model_path else None,
    )
    stored_diff_config_path = _safe_copy(
        _uploaded_path_obj(diff_config_path),
        local_model_dir / _uploaded_display_name(diff_config_path) if diff_config_path else None,
    )

    return (
        stored_model_path,
        stored_config_path,
        stored_cluster_path,
        stored_diff_model_path,
        stored_diff_config_path,
    )


def _resolve_local_model_dir(local_model_selection: str, local_model_checkpoint_selection: str | None = None):
    model_dir = Path(local_model_selection)
    if not model_dir.exists():
        raise FileNotFoundError(f"本地模型目录不存在：{model_dir}")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"本地模型目录缺少 config.json：{model_dir}")

    generator_ckpts = sorted(model_dir.glob("G_*.pth"), key=_checkpoint_step)
    if not generator_ckpts:
        raise FileNotFoundError(f"本地模型目录缺少 G_*.pth：{model_dir}")

    selected_checkpoint = (local_model_checkpoint_selection or "").strip()
    if selected_checkpoint:
        checkpoint_path = Path(selected_checkpoint)
        if checkpoint_path not in generator_ckpts:
            raise FileNotFoundError(f"所选 G_*.pth 不属于当前本地模型目录：{checkpoint_path}")
    else:
        checkpoint_path = generator_ckpts[-1]

    return str(checkpoint_path), str(config_path)


def resolve_model_inputs(model_path, config_path, cluster_model_path, diff_model_path, diff_config_path, local_model_enabled, local_model_selection, local_model_checkpoint_selection):
    if local_model_enabled:
        resolved_model_path, resolved_config_path = _resolve_local_model_dir(local_model_selection, local_model_checkpoint_selection)
        resolved_cluster_model_path = resolve_uploaded_path(cluster_model_path)
        resolved_diff_model_path = resolve_uploaded_path(diff_model_path)
        resolved_diff_config_path = resolve_uploaded_path(diff_config_path)
    else:
        (
            resolved_model_path,
            resolved_config_path,
            resolved_cluster_model_path,
            resolved_diff_model_path,
            resolved_diff_config_path,
        ) = _store_uploaded_model_bundle(
            model_path,
            config_path,
            cluster_model_path,
            diff_model_path,
            diff_config_path,
        )

    return (
        resolved_model_path,
        resolved_config_path,
        resolved_cluster_model_path,
        resolved_diff_model_path,
        resolved_diff_config_path,
    )
