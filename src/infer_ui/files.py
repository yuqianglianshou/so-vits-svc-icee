from __future__ import annotations

from pathlib import Path


def resolve_uploaded_path(file_obj):
    if file_obj is None:
        return ""
    return getattr(file_obj, "name", file_obj)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (TypeError, ValueError):
        return -1


def _resolve_local_model_dir(local_model_selection: str):
    model_dir = Path(local_model_selection)
    if not model_dir.exists():
        raise FileNotFoundError(f"本地模型目录不存在：{model_dir}")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"本地模型目录缺少 config.json：{model_dir}")

    generator_ckpts = sorted(model_dir.glob("G_*.pth"), key=_checkpoint_step)
    if not generator_ckpts:
        raise FileNotFoundError(f"本地模型目录缺少 G_*.pth：{model_dir}")

    return str(generator_ckpts[-1]), str(config_path)


def resolve_model_inputs(model_path, config_path, cluster_model_path, diff_model_path, diff_config_path, local_model_enabled, local_model_selection):
    if local_model_enabled:
        resolved_model_path, resolved_config_path = _resolve_local_model_dir(local_model_selection)
    else:
        resolved_model_path = resolve_uploaded_path(model_path)
        resolved_config_path = resolve_uploaded_path(config_path)

    return (
        resolved_model_path,
        resolved_config_path,
        resolve_uploaded_path(cluster_model_path),
        resolve_uploaded_path(diff_model_path),
        resolve_uploaded_path(diff_config_path),
    )
