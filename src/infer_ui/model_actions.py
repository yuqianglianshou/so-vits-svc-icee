from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

import gradio as gr

from src.infer_ui.bundle import export_local_model_bundle, inspect_infer_bundle_extras


def export_local_model_bundle_action(model_selection, model_checkpoint_selection):
    bundle_path, bundle_meta = export_local_model_bundle(model_selection, model_checkpoint_selection)
    extras = []
    if bundle_meta.get("has_diffusion"):
        extras.append("扩散")
    if bundle_meta.get("has_index"):
        extras.append("索引")
    if bundle_meta.get("has_cover"):
        extras.append("封面")
    if bundle_meta.get("has_description"):
        extras.append("说明")
    extras_text = f"；已自动包含：{', '.join(extras)}" if extras else "；当前目录未检测到可打包的额外资产"
    return (
        f'<div class="subtle-note">已导出当前模型包{extras_text}</div>',
        bundle_path.as_posix(),
        gr.update(interactive=True),
    )


def open_exported_file_location(exported_path, debug: bool = False):
    if not exported_path:
        raise gr.Error("当前还没有可打开位置的导出文件。")

    target = Path(exported_path).expanduser().resolve()
    if not target.exists():
        raise gr.Error(f"导出文件不存在：{target.as_posix()}")

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["/usr/bin/open", "-R", str(target)], start_new_session=True)
        elif os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(target)])
        else:
            subprocess.Popen(["/usr/bin/xdg-open", str(target.parent)], start_new_session=True)
        return gr.update()
    except Exception as exc:
        if debug:
            traceback.print_exc()
        raise gr.Error(f"无法打开文件位置：{exc}")


def build_extra_option_updates(*, has_diffusion: bool, has_cluster: bool):
    return (
        gr.update(value=has_diffusion, interactive=has_diffusion),
        gr.update(value=has_cluster, interactive=has_cluster),
    )


def uploaded_bundle_option_refresh_fn(model_path):
    bundle_path = getattr(model_path, "name", model_path) if model_path is not None else ""
    extras = inspect_infer_bundle_extras(bundle_path)
    return build_extra_option_updates(
        has_diffusion=bool(extras.get("has_diffusion")),
        has_cluster=bool(extras.get("has_index")),
    )
