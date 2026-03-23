from __future__ import annotations
"""推理页文件输入与路径解析辅助函数。

这里统一处理上传文件值、本地模型目录和增强文件路径解析，
避免上传模式与本地模式的判断逻辑散落在主页面里。
"""

import glob
import os


def resolve_uploaded_path(file_obj):
    """把 Gradio 上传值统一转换成可用的本地路径字符串。"""
    if file_obj is None:
        return ""
    return getattr(file_obj, "name", file_obj)



def resolve_model_inputs(model_path, config_path, cluster_model_path, diff_model_path, diff_config_path, local_model_enabled, local_model_selection):
    """统一解析上传模式和本地模式下的模型相关文件路径。"""
    if local_model_enabled:
        resolved_model_path = glob.glob(os.path.join(local_model_selection, '*.pth'))[0]
        resolved_config_path = glob.glob(os.path.join(local_model_selection, '*.json'))[0]
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
