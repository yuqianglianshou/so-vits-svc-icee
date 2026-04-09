from __future__ import annotations

from src.infer_ui.text import build_runtime_summary, render_load_result_html


def resolve_device_choice(device_label, cuda_map):
    if isinstance(device_label, str) and "CUDA" in device_label:
        return cuda_map[device_label]
    return device_label


def get_model_device_name(model):
    device = getattr(model, "dev", None)
    if device is None:
        return "unknown"

    device_text = str(device)
    if "cuda" not in device_text:
        return device_text

    properties = getattr(device, "properties", None)
    if properties is not None and getattr(properties, "name", None):
        return properties.name

    try:
        import torch

        index = getattr(device, "index", None)
        if index is None:
            index = torch.cuda.current_device()
        return torch.cuda.get_device_name(index)
    except Exception:
        return device_text


def build_loaded_model_result(model, cluster_model_path, diff_model_path):
    spks = list(model.spk2id.keys())
    primary_spk = spks[0]
    cluster_status = "未加载"
    if cluster_model_path:
        cluster_status = f"已加载{'音色增强' if model.feature_retrieval else '聚类增强'}"
    diffusion_status = "已加载" if diff_model_path else "未加载"
    quality_hint = "当前已接近推荐高质量链路。"
    if not diff_model_path and not cluster_model_path:
        quality_hint = "当前只加载了基础模型，可以正常转换，但不是最佳效果。建议补充音质增强模型和音色增强文件。"
    elif not diff_model_path:
        quality_hint = "当前缺少音质增强模型，可以正常转换，但音质上限还没拉满。"
    elif not cluster_model_path:
        quality_hint = "当前缺少音色增强文件，音色相似度还有提升空间。"
    summary = (
        f"设备：{get_model_device_name(model)}\n"
        "So-VITS：已加载\n"
        f"音质增强：{diffusion_status}\n"
        f"音色增强：{cluster_status}\n"
        f"当前模型音色：{primary_spk}\n"
        f"质量提示：{quality_hint}"
    )
    return [primary_spk], primary_spk, render_load_result_html(summary)


def export_runtime_summary_for_model(model, sid, quality_mode, vc_transform, cluster_ratio, k_step):
    has_diffusion = bool(getattr(model, "shallow_diffusion", False) or getattr(model, "only_diffusion", False))
    has_cluster = getattr(model, "cluster_model", None) is not None
    speaker_name = sid or "未选择"
    return build_runtime_summary(
        get_model_device_name(model),
        speaker_name,
        quality_mode,
        vc_transform,
        cluster_ratio,
        k_step,
        has_diffusion,
        has_cluster,
    )
