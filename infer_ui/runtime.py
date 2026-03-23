from __future__ import annotations
"""推理页模型运行态与摘要辅助函数。

这里放“模型已经加载之后”才会用到的设备、状态和摘要逻辑，
便于把页面渲染和推理执行过程解耦。
"""

from infer_ui.text import build_runtime_summary, render_load_result_html



def resolve_device_choice(device_label, cuda_map):
    """把界面设备选项转换成 Svc 可用的设备参数。"""
    if isinstance(device_label, str) and "CUDA" in device_label:
        return cuda_map[device_label]
    return device_label



def get_model_device_name(model):
    """返回当前已加载模型的设备名称。"""
    return model.dev.properties.name if "cuda" in str(model.dev) else str(model.dev)



def build_loaded_model_result(model, cluster_model_path, diff_model_path):
    """根据当前已加载模型构造加载结果和音色下拉默认值。"""
    spks = list(model.spk2id.keys())
    cluster_status = "未加载"
    if cluster_model_path:
        cluster_status = f"已加载 {'音色增强' if model.feature_retrieval else '聚类增强'}"
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
        f"音色数量：{len(spks)}\n"
        f"默认音色：{spks[0]}\n"
        f"质量提示：{quality_hint}"
    )
    return spks, spks[0], render_load_result_html(summary)



def export_runtime_summary_for_model(model, sid, quality_mode, vc_transform, cluster_ratio, k_step):
    """为当前已加载模型生成运行摘要文本。"""
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
