from __future__ import annotations

from dataclasses import dataclass

from src.infer_ui.local_models import (
    detect_local_model_extras,
    list_local_model_checkpoints,
    list_local_model_diffusion_checkpoints,
    load_last_selected_infer_model,
    scan_imported_models,
    scan_workspace_models,
)


@dataclass
class InitialModelState:
    workspace_choices: list[str]
    imported_choices: list[str]
    model_source: str
    workspace_selection: str | None
    imported_selection: str | None
    workspace_checkpoints: list[str]
    imported_checkpoints: list[str]
    workspace_checkpoint_selection: str | None
    imported_checkpoint_selection: str | None
    workspace_diffusion_checkpoints: list[str]
    imported_diffusion_checkpoints: list[str]
    workspace_diffusion_checkpoint_selection: str | None
    imported_diffusion_checkpoint_selection: str | None
    local_model_enabled: bool
    selected_model_dir: str | None
    diff_model_path: str
    diff_config_path: str
    cluster_model_path: str
    workspace_diff_model_path: str
    workspace_diff_config_path: str
    workspace_cluster_model_path: str
    imported_diff_model_path: str
    imported_diff_config_path: str
    imported_cluster_model_path: str
    has_diffusion: bool
    has_cluster: bool


def build_initial_model_state() -> InitialModelState:
    workspace_choices = scan_workspace_models()
    imported_choices = scan_imported_models()
    remembered_source, remembered_model_dir = load_last_selected_infer_model()

    if workspace_choices:
        model_source = "workspace"
    elif imported_choices:
        model_source = "imported"
    else:
        model_source = "workspace"

    workspace_selection = (
        remembered_model_dir
        if model_source == "workspace" and remembered_model_dir in workspace_choices
        else (workspace_choices[0] if workspace_choices else None)
    )
    imported_selection = (
        remembered_model_dir
        if model_source == "imported" and remembered_model_dir in imported_choices
        else (imported_choices[0] if imported_choices else None)
    )

    workspace_checkpoints = list_local_model_checkpoints(workspace_selection) if workspace_selection else []
    imported_checkpoints = list_local_model_checkpoints(imported_selection) if imported_selection else []
    workspace_checkpoint_selection = workspace_checkpoints[-1] if workspace_checkpoints else None
    imported_checkpoint_selection = imported_checkpoints[-1] if imported_checkpoints else None

    workspace_diffusion_checkpoints = list_local_model_diffusion_checkpoints(workspace_selection) if workspace_selection else []
    imported_diffusion_checkpoints = list_local_model_diffusion_checkpoints(imported_selection) if imported_selection else []
    workspace_diffusion_checkpoint_selection = workspace_diffusion_checkpoints[-1] if workspace_diffusion_checkpoints else None
    imported_diffusion_checkpoint_selection = imported_diffusion_checkpoints[-1] if imported_diffusion_checkpoints else None

    local_model_enabled = bool(workspace_selection or imported_selection)
    selected_model_dir = workspace_selection if model_source == "workspace" else imported_selection

    diff_model_path, diff_config_path, cluster_model_path = detect_local_model_extras(selected_model_dir)
    workspace_diff_model_path, workspace_diff_config_path, workspace_cluster_model_path = detect_local_model_extras(workspace_selection)
    imported_diff_model_path, imported_diff_config_path, imported_cluster_model_path = detect_local_model_extras(imported_selection)

    return InitialModelState(
        workspace_choices=workspace_choices,
        imported_choices=imported_choices,
        model_source=model_source,
        workspace_selection=workspace_selection,
        imported_selection=imported_selection,
        workspace_checkpoints=workspace_checkpoints,
        imported_checkpoints=imported_checkpoints,
        workspace_checkpoint_selection=workspace_checkpoint_selection,
        imported_checkpoint_selection=imported_checkpoint_selection,
        workspace_diffusion_checkpoints=workspace_diffusion_checkpoints,
        imported_diffusion_checkpoints=imported_diffusion_checkpoints,
        workspace_diffusion_checkpoint_selection=workspace_diffusion_checkpoint_selection,
        imported_diffusion_checkpoint_selection=imported_diffusion_checkpoint_selection,
        local_model_enabled=local_model_enabled,
        selected_model_dir=selected_model_dir,
        diff_model_path=diff_model_path,
        diff_config_path=diff_config_path,
        cluster_model_path=cluster_model_path,
        workspace_diff_model_path=workspace_diff_model_path,
        workspace_diff_config_path=workspace_diff_config_path,
        workspace_cluster_model_path=workspace_cluster_model_path,
        imported_diff_model_path=imported_diff_model_path,
        imported_diff_config_path=imported_diff_config_path,
        imported_cluster_model_path=imported_cluster_model_path,
        has_diffusion=bool(diff_model_path and diff_config_path),
        has_cluster=bool(cluster_model_path),
    )
