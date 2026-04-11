from __future__ import annotations

from pathlib import Path

from src.infer_ui.bundle import unpack_infer_bundle
from src.infer_ui.local_models import detect_local_model_extras, list_local_model_diffusion_checkpoints


def resolve_uploaded_path(file_obj):
    if file_obj is None:
        return ""
    return getattr(file_obj, "name", file_obj)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (TypeError, ValueError):
        return -1


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


def _resolve_selected_library_model(model_source, workspace_model_selection, workspace_model_checkpoint_selection, imported_model_selection, imported_model_checkpoint_selection):
    if model_source == "workspace":
        return _resolve_local_model_dir(workspace_model_selection, workspace_model_checkpoint_selection), workspace_model_selection
    return _resolve_local_model_dir(imported_model_selection, imported_model_checkpoint_selection), imported_model_selection


def _resolve_local_diffusion_model(model_dir: str, local_diffusion_checkpoint_selection: str | None = None):
    diff_model_path, diff_config_path, _ = detect_local_model_extras(model_dir)
    diffusion_ckpts = [Path(path) for path in list_local_model_diffusion_checkpoints(model_dir)]
    selected_checkpoint = (local_diffusion_checkpoint_selection or "").strip()
    if selected_checkpoint:
        checkpoint_path = Path(selected_checkpoint)
        if checkpoint_path not in diffusion_ckpts:
            raise FileNotFoundError(f"所选音质增强模型 `.pt` 不属于当前模型目录：{checkpoint_path}")
        diff_model_path = str(checkpoint_path)
    return diff_model_path, diff_config_path


def resolve_model_inputs(model_path, config_path, cluster_model_path, diff_model_path, diff_config_path, local_model_enabled, model_source, workspace_model_selection, workspace_model_checkpoint_selection, workspace_diffusion_checkpoint_selection, imported_model_selection, imported_model_checkpoint_selection, imported_diffusion_checkpoint_selection, enable_diffusion=True, enable_cluster=True):
    if local_model_enabled:
        (resolved_model_path, resolved_config_path), selected_model_dir = _resolve_selected_library_model(
            model_source,
            workspace_model_selection,
            workspace_model_checkpoint_selection,
            imported_model_selection,
            imported_model_checkpoint_selection,
        )
        local_diffusion_checkpoint_selection = workspace_diffusion_checkpoint_selection if model_source == "workspace" else imported_diffusion_checkpoint_selection
        resolved_diff_model_path, resolved_diff_config_path = _resolve_local_diffusion_model(
            selected_model_dir,
            local_diffusion_checkpoint_selection,
        )
        _, _, resolved_cluster_model_path = detect_local_model_extras(selected_model_dir)
    else:
        bundle_path = resolve_uploaded_path(model_path)
        if not bundle_path:
            raise FileNotFoundError("手动上传模式需要选择单文件推理包。")
        (
            resolved_model_path,
            resolved_config_path,
            _,
            bundled_diff_model_path,
            bundled_diff_config_path,
            bundled_cluster_model_path,
        ) = unpack_infer_bundle(bundle_path)
        resolved_cluster_model_path = bundled_cluster_model_path
        resolved_diff_model_path = bundled_diff_model_path
        resolved_diff_config_path = bundled_diff_config_path

    if not enable_diffusion:
        resolved_diff_model_path = ""
        resolved_diff_config_path = ""
    if not enable_cluster:
        resolved_cluster_model_path = ""

    return (
        resolved_model_path,
        resolved_config_path,
        resolved_cluster_model_path,
        resolved_diff_model_path,
        resolved_diff_config_path,
    )
