from __future__ import annotations

import os
import traceback
from pathlib import Path

import gradio as gr
import torch

from src.inference.infer_tool import Svc
from src.infer_ui.files import resolve_model_inputs
from src.infer_ui.local_models import save_last_selected_infer_model
from src.infer_ui.runtime import (
    build_loaded_model_result,
    export_runtime_summary_for_model,
    render_load_result_html,
    resolve_device_choice,
)


def load_model_from_paths(model_path, config_path, cluster_model_path, device, enhance, diff_model_path, diff_config_path, only_diffusion, cuda_map):
    device = resolve_device_choice(device, cuda_map)
    cluster_filepath = os.path.split(cluster_model_path) if cluster_model_path else ("", "no_cluster")
    feature_retrieval = ".pkl" in cluster_filepath[1]
    model = Svc(
        model_path,
        config_path,
        device=device if device != "Auto" else None,
        cluster_model_path=cluster_model_path or "",
        nsf_hifigan_enhance=enhance,
        diffusion_model_path=diff_model_path or "",
        diffusion_config_path=diff_config_path or "",
        shallow_diffusion=True if diff_model_path else False,
        only_diffusion=only_diffusion,
        feature_retrieval=feature_retrieval,
    )
    sid_update, default_spk, summary_html = build_loaded_model_result(model, cluster_model_path, diff_model_path)
    return model, gr.update(choices=sid_update, value=default_spk), summary_html


def analyze_and_load_model(
    model_path,
    config_path,
    cluster_model_path,
    device,
    enhance,
    diff_model_path,
    diff_config_path,
    only_diffusion,
    local_model_enabled,
    model_source,
    workspace_model_selection,
    workspace_model_checkpoint_selection,
    workspace_diffusion_checkpoint_selection,
    imported_model_selection,
    imported_model_checkpoint_selection,
    imported_diffusion_checkpoint_selection,
    enable_diffusion,
    enable_cluster,
    cuda_map,
):
    model_path, config_path, cluster_model_path, diff_model_path, diff_config_path = resolve_model_inputs(
        model_path,
        config_path,
        cluster_model_path,
        diff_model_path,
        diff_config_path,
        local_model_enabled,
        model_source,
        workspace_model_selection,
        workspace_model_checkpoint_selection,
        workspace_diffusion_checkpoint_selection,
        imported_model_selection,
        imported_model_checkpoint_selection,
        imported_diffusion_checkpoint_selection,
        enable_diffusion=enable_diffusion,
        enable_cluster=enable_cluster,
    )
    if local_model_enabled:
        save_last_selected_infer_model(model_source or "imported", str(Path(model_path).parent))
    return load_model_from_paths(
        model_path,
        config_path,
        cluster_model_path,
        device,
        enhance,
        diff_model_path,
        diff_config_path,
        only_diffusion,
        cuda_map,
    )


def safe_analyze_and_load_model(*args, debug: bool = False, **kwargs):
    try:
        return analyze_and_load_model(*args, **kwargs)
    except Exception as exc:
        if debug:
            traceback.print_exc()
        raise gr.Error(exc)


def unload_model(current_model):
    if current_model is None:
        return None, gr.update(choices=[], value=""), render_load_result_html("当前没有已加载模型。")
    current_model.unload_model()
    torch.cuda.empty_cache()
    return None, gr.update(choices=[], value=""), render_load_result_html("模型已卸载。")


def export_runtime_summary(current_model, sid, quality_mode, vc_transform, cluster_ratio, k_step):
    if current_model is None:
        return "当前没有已加载模型，无法生成运行摘要。"
    return export_runtime_summary_for_model(current_model, sid, quality_mode, vc_transform, cluster_ratio, k_step)
