import logging
import os
import socket
import subprocess
import sys
import traceback
import webbrowser
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch

from src.gradio_api_info_fallback import apply_gradio_4_api_info_patch
from src.inference.infer_tool import Svc
from src.infer_ui.local_models import (
    IMPORTED_MODEL_ROOT,
    WORKSPACE_MODEL_ROOT,
    detect_local_model_extras,
    imported_model_refresh_fn,
    list_local_model_checkpoints,
    load_last_selected_infer_model,
    model_checkpoint_refresh_fn,
    model_extra_refresh_fn,
    model_option_refresh_fn,
    persist_selected_model,
    save_last_selected_infer_model,
    scan_imported_models,
    scan_workspace_models,
    workspace_model_refresh_fn,
)
from src.infer_ui.bundle import BUNDLE_VERSION, export_local_model_bundle, inspect_infer_bundle_extras
from src.infer_ui.convert import quality_convert
from src.infer_ui.files import resolve_model_inputs
from src.infer_ui.runtime import (
    build_loaded_model_result,
    export_runtime_summary_for_model,
    resolve_device_choice,
)
from src.infer_ui.text import (
    render_convert_result_html,
    render_load_result_html,
)
from src.quality_presets import BEST_QUALITY_PRESET, QUALITY_MODES

apply_gradio_4_api_info_patch()

CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent
INFER_PAGE_CSS = (CODE_ROOT / "infer_ui" / "page.css").read_text(encoding="utf-8")


def find_available_port(default_port: int, max_tries: int = 20):
    """为推理页寻找一个可用端口，优先尊重环境变量覆盖。"""
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass

    for offset in range(max_tries):
        port = default_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        fallback_port = sock.getsockname()[1]
        if fallback_port:
            return fallback_port
    raise OSError(f"无法在 {default_port}-{default_port + max_tries - 1} 范围内找到可用端口。")

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
debug = False

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

def load_model_from_paths(model_path, config_path, cluster_model_path, device, enhance, diff_model_path, diff_config_path, only_diffusion):
    """按给定文件路径加载 So-VITS 模型，并返回音色列表与加载摘要。"""
    global model
    try:
        device = resolve_device_choice(device, cuda)
        cluster_filepath = os.path.split(cluster_model_path) if cluster_model_path else ("", "no_cluster")
        fr = ".pkl" in cluster_filepath[1]
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
            feature_retrieval=fr,
        )
        spks, default_spk, summary_html = build_loaded_model_result(model, cluster_model_path, diff_model_path)
        return gr.update(choices=spks, value=default_spk), summary_html
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def modelAnalysis(model_path, config_path, cluster_model_path, device, enhance, diff_model_path, diff_config_path, only_diffusion, local_model_enabled, model_source, workspace_model_selection, workspace_model_checkpoint_selection, imported_model_selection, imported_model_checkpoint_selection, enable_diffusion, enable_cluster):
    """统一处理上传模式和本地模式下的模型加载入口。"""
    try:
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
            imported_model_selection,
            imported_model_checkpoint_selection,
            enable_diffusion=enable_diffusion,
            enable_cluster=enable_cluster,
        )
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
        )
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def export_runtime_summary(sid, quality_mode, vc_transform, cluster_ratio, k_step):
    """导出当前已加载模型的推理运行摘要。"""
    global model
    if model is None:
        return "当前没有已加载模型，无法生成运行摘要。"
    return export_runtime_summary_for_model(model, sid, quality_mode, vc_transform, cluster_ratio, k_step)

    
def modelUnload():
    """卸载当前模型，并清理显存缓存。"""
    global model
    if model is None:
        return gr.update(choices=[], value=""), render_load_result_html("当前没有已加载模型。")
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return gr.update(choices=[], value=""), render_load_result_html("模型已卸载。")

def apply_quality_mode(mode):
    """根据质量模式同步推荐的检索比例和浅扩散步数。"""
    preset = QUALITY_MODES[mode]
    return preset["cluster_ratio"], preset["k_step"]


def quality_vc_fn(sid, input_audio, quality_mode, vc_transform, cluster_ratio, k_step):
    """推理页主入口：按当前高质量预设执行一次转换。"""
    global model
    return quality_convert(model, sid, input_audio, quality_mode, vc_transform, cluster_ratio, k_step, BEST_QUALITY_PRESET)


def describe_input_audio(audio_path):
    if not audio_path:
        return '<div class="subtle-note">尚未选择输入音频。</div>'
    try:
        info = sf.info(audio_path)
        minutes = int(info.duration // 60)
        seconds = int(round(info.duration % 60))
        if seconds == 60:
            minutes += 1
            seconds = 0
        duration_text = f"{minutes}:{seconds:02d}"
        return (
            '<div class="subtle-note">'
            f'后端识别时长：{duration_text}（{info.duration:.2f} 秒）；'
            f'采样率：{info.samplerate} Hz；'
            f'声道：{info.channels}；'
            f'格式：{info.format}'
            '</div>'
        )
    except Exception as exc:
        return f'<div class="subtle-note">无法读取输入音频信息：{type(exc).__name__}: {exc}</div>'


def preview_input_audio(audio_path):
    return audio_path or None


def export_local_model_bundle_action(model_selection, model_checkpoint_selection):
    try:
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
    except Exception as exc:
        if debug:
            traceback.print_exc()
        raise gr.Error(exc)


def open_exported_file_location(exported_path):
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


with gr.Blocks(
    analytics_enabled=False,
    theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
        font=["Avenir Next", "PingFang SC", "Helvetica Neue", "sans-serif"],
        font_mono=['JetBrains Mono', "SFMono-Regular", 'Consolas']
    ),
    css=INFER_PAGE_CSS,
) as app:
    initial_workspace_choices = scan_workspace_models()
    initial_imported_choices = scan_imported_models()
    remembered_source, remembered_model_dir = load_last_selected_infer_model()
    if initial_workspace_choices:
        initial_model_source = "workspace"
    elif initial_imported_choices:
        initial_model_source = "imported"
    else:
        initial_model_source = "workspace"

    initial_workspace_selection = (
        remembered_model_dir
        if initial_model_source == "workspace" and remembered_model_dir in initial_workspace_choices
        else (initial_workspace_choices[0] if initial_workspace_choices else None)
    )
    initial_imported_selection = (
        remembered_model_dir
        if initial_model_source == "imported" and remembered_model_dir in initial_imported_choices
        else (initial_imported_choices[0] if initial_imported_choices else None)
    )
    initial_workspace_checkpoints = list_local_model_checkpoints(initial_workspace_selection) if initial_workspace_selection else []
    initial_imported_checkpoints = list_local_model_checkpoints(initial_imported_selection) if initial_imported_selection else []
    initial_workspace_checkpoint_selection = initial_workspace_checkpoints[-1] if initial_workspace_checkpoints else None
    initial_imported_checkpoint_selection = initial_imported_checkpoints[-1] if initial_imported_checkpoints else None
    initial_local_model_enabled = bool(initial_workspace_selection or initial_imported_selection)
    initial_selected_model_dir = initial_workspace_selection if initial_model_source == "workspace" else initial_imported_selection
    initial_diff_model_path, initial_diff_config_path, initial_cluster_model_path = detect_local_model_extras(initial_selected_model_dir)
    initial_workspace_diff_model_path, initial_workspace_diff_config_path, initial_workspace_cluster_model_path = detect_local_model_extras(initial_workspace_selection)
    initial_imported_diff_model_path, initial_imported_diff_config_path, initial_imported_cluster_model_path = detect_local_model_extras(initial_imported_selection)
    initial_has_diffusion = bool(initial_diff_model_path and initial_diff_config_path)
    initial_has_cluster = bool(initial_cluster_model_path)

    gr.HTML("""
        <div class="hero">
            <h2>高音质唱歌转换</h2>
        </div>
    """)

    local_model_enabled = gr.Checkbox(value=initial_local_model_enabled, visible=False)
    model_source = gr.State(value=initial_model_source)
    enhance = gr.Checkbox(value=False, visible=False)
    only_diffusion = gr.Checkbox(value=False, visible=False)
    with gr.Row():
        with gr.Column(scale=6, elem_classes=["step-card", "card-model"]):
            gr.Markdown("### 1. 模型文件")
            gr.Markdown("先选训练工作区模型、已导入的模型，或上传单文件推理包。")
            with gr.Tabs():
                with gr.TabItem('训练工作区') as workspace_model_tab:
                    gr.Markdown(f"训练工作区目录：`{WORKSPACE_MODEL_ROOT}`")
                    workspace_model_refresh_btn = gr.Button('刷新训练工作区列表', elem_classes=["info-action"], elem_id="infer-workspace-refresh")
                    workspace_model_selection = gr.Dropdown(
                        label='选择训练工作区模型',
                        choices=initial_workspace_choices,
                        value=initial_workspace_selection,
                        interactive=True,
                    )
                    workspace_model_checkpoint_selection = gr.Dropdown(
                        label='选择主模型 G_*.pth',
                        choices=initial_workspace_checkpoints,
                        value=initial_workspace_checkpoint_selection,
                        interactive=bool(initial_workspace_checkpoints),
                    )
                    with gr.Accordion("音质增强和音色增强文件", open=True):
                        with gr.Row():
                            diff_model_path = gr.Textbox(label="音质增强模型 `.pt`", value=initial_workspace_diff_model_path or "", interactive=False)
                            diff_config_path = gr.Textbox(label="音质增强配置 `.yaml`", value=initial_workspace_diff_config_path or "", interactive=False)
                        cluster_model_path = gr.Textbox(label="音色增强索引 `.pkl`", value=initial_workspace_cluster_model_path or "", interactive=False)
                    workspace_export_bundle_btn = gr.Button("导出当前模型包", elem_classes=["info-action"])
                    workspace_export_bundle_status = gr.HTML('<div class="subtle-note">会基于当前选中的 G_*.pth 导出当前模型包；若目录里存在扩散、索引，也会自动一起打包。</div>')
                    workspace_export_bundle_path = gr.Textbox(label="生成路径", value="", interactive=False)
                    workspace_open_export_location_btn = gr.Button("打开文件位置", interactive=False, elem_classes=["info-action"])
                with gr.TabItem('已导入的模型') as imported_model_tab:
                    gr.Markdown(f"已导入模型目录：`{IMPORTED_MODEL_ROOT}`")
                    imported_model_refresh_btn = gr.Button('刷新已导入模型列表', elem_classes=["info-action"], elem_id="infer-imported-refresh")
                    imported_model_selection = gr.Dropdown(
                        label='选择已导入的模型',
                        choices=initial_imported_choices,
                        value=initial_imported_selection,
                        interactive=True,
                    )
                    imported_model_checkpoint_selection = gr.Dropdown(
                        label='选择主模型 G_*.pth',
                        choices=initial_imported_checkpoints,
                        value=initial_imported_checkpoint_selection,
                        interactive=bool(initial_imported_checkpoints),
                    )
                    with gr.Accordion("音质增强和音色增强文件", open=True):
                        with gr.Row():
                            imported_diff_model_path = gr.Textbox(label="音质增强模型 `.pt`", value=initial_imported_diff_model_path or "", interactive=False)
                            imported_diff_config_path = gr.Textbox(label="音质增强配置 `.yaml`", value=initial_imported_diff_config_path or "", interactive=False)
                        imported_cluster_model_path = gr.Textbox(label="音色增强索引 `.pkl`", value=initial_imported_cluster_model_path or "", interactive=False)
                    imported_export_bundle_btn = gr.Button("导出当前模型包", elem_classes=["info-action"])
                    imported_export_bundle_status = gr.HTML('<div class="subtle-note">会基于当前选中的 G_*.pth 导出当前模型包；若目录里存在扩散、索引，也会自动一起打包。</div>')
                    imported_export_bundle_path = gr.Textbox(label="生成路径", value="", interactive=False)
                    imported_open_export_location_btn = gr.Button("打开文件位置", interactive=False, elem_classes=["info-action"])
                with gr.TabItem('手动上传') as local_model_tab_upload:
                    model_path = gr.File(label="单文件推理包 `.pth`")
                    config_path = gr.State(value=None)
        with gr.Column(scale=5, elem_classes=["step-card", "card-status"]):
            gr.Markdown("### 2. 加载与确认")
            device = gr.Dropdown(label="推理设备", choices=["Auto", *cuda.keys(), "cpu"], value="Auto")
            with gr.Row():
                enable_diffusion = gr.Checkbox(label="音质增强", value=initial_has_diffusion, interactive=initial_has_diffusion)
                enable_cluster = gr.Checkbox(label="音色增强", value=initial_has_cluster, interactive=initial_has_cluster)
            model_load_button = gr.Button(value="加载模型", variant="primary", elem_classes=["primary-action"])
            model_unload_button = gr.Button(value="卸载模型", elem_classes=["danger-secondary"], elem_id="infer-unload")
            sid = gr.Dropdown(label="当前模型音色")
            gr.Markdown("#### 加载结果")
            sid_output = gr.HTML(render_load_result_html("先加载主模型包。"))

    with gr.Row():
        with gr.Column(scale=5, elem_classes=["step-card", "card-params"]):
            gr.Markdown("### 3. 高音质参数")
            gr.Markdown("通常只需要选质量模式，再改 `变调`。")
            quality_mode = gr.Radio(
                label="质量模式",
                choices=["标准高质", "极致质量"],
                value="极致质量",
            )
            vc_transform = gr.Number(label="变调（半音）", value=0)
            cluster_ratio = gr.Slider(label="特征检索混合比例", minimum=0, maximum=1, step=0.05, value=QUALITY_MODES["极致质量"]["cluster_ratio"])
            k_step = gr.Slider(label="浅扩散步数", minimum=1, maximum=1000, step=1, value=QUALITY_MODES["极致质量"]["k_step"])
            best_quality_preset_btn = gr.Button(value="恢复推荐高音质参数", elem_classes=["success-action"], elem_id="infer-preset")
            export_summary_btn = gr.Button(value="生成运行摘要", elem_classes=["info-action"], elem_id="infer-summary")
            runtime_summary_output = gr.Textbox(
                label="运行摘要",
                elem_classes=["compact-status"],
                value="加载模型后可生成。",
            )
            with gr.Accordion("默认参数说明", open=False):
                gr.Markdown("""
                - `标准高质`：先试听
                - `极致质量`：更慢但更接近上限
                - 输出固定为 `flac`
                - F0 预测固定为 `rmvpe`
                - 其余参数已按高质量预设固定
                """)
        with gr.Column(scale=6, elem_classes=["step-card", "card-output"]):
            gr.Markdown("### 4. 上传音频并转换")
            vc_input3 = gr.File(
                label="选择输入音频文件",
                file_types=["audio"],
                type="filepath",
            )
            input_audio_info = gr.HTML('<div class="subtle-note">尚未选择输入音频。</div>')
            vc_input_preview = gr.Audio(label="输入音频", interactive=False, elem_classes=["result-audio"])
            vc_submit = gr.Button("开始转换", variant="primary", elem_classes=["primary-action"])
            gr.Markdown("#### 处理结果")
            vc_output1 = gr.HTML(render_convert_result_html("等待开始转换。"))
            vc_output2 = gr.Audio(label="输出音频", interactive=False, elem_classes=["result-audio"])

    workspace_model_refresh_btn.click(workspace_model_refresh_fn, outputs=workspace_model_selection, show_api=False).then(
        model_checkpoint_refresh_fn,
        [workspace_model_selection],
        [workspace_model_checkpoint_selection],
        show_api=False,
    ).then(
        model_option_refresh_fn,
        [workspace_model_selection],
        [enable_diffusion, enable_cluster],
        show_api=False,
    ).then(
        model_extra_refresh_fn,
        [workspace_model_selection],
        [diff_model_path, diff_config_path, cluster_model_path],
        show_api=False,
    )
    imported_model_refresh_btn.click(imported_model_refresh_fn, outputs=imported_model_selection, show_api=False).then(
        model_checkpoint_refresh_fn,
        [imported_model_selection],
        [imported_model_checkpoint_selection],
        show_api=False,
    ).then(
        model_option_refresh_fn,
        [imported_model_selection],
        [enable_diffusion, enable_cluster],
        show_api=False,
    ).then(
        model_extra_refresh_fn,
        [imported_model_selection],
        [imported_diff_model_path, imported_diff_config_path, imported_cluster_model_path],
        show_api=False,
    )
    local_model_tab_upload.select(lambda: False, outputs=local_model_enabled, show_api=False)
    local_model_tab_upload.select(
        uploaded_bundle_option_refresh_fn,
        [model_path],
        [enable_diffusion, enable_cluster],
        show_api=False,
    )
    workspace_model_tab.select(lambda: True, outputs=local_model_enabled, show_api=False)
    workspace_model_tab.select(lambda: "workspace", outputs=model_source, show_api=False)
    workspace_model_tab.select(
        model_option_refresh_fn,
        [workspace_model_selection],
        [enable_diffusion, enable_cluster],
        show_api=False,
    )
    workspace_model_tab.select(
        model_extra_refresh_fn,
        [workspace_model_selection],
        [diff_model_path, diff_config_path, cluster_model_path],
        show_api=False,
    )
    imported_model_tab.select(lambda: True, outputs=local_model_enabled, show_api=False)
    imported_model_tab.select(lambda: "imported", outputs=model_source, show_api=False)
    imported_model_tab.select(
        model_option_refresh_fn,
        [imported_model_selection],
        [enable_diffusion, enable_cluster],
        show_api=False,
    )
    imported_model_tab.select(
        model_extra_refresh_fn,
        [imported_model_selection],
        [imported_diff_model_path, imported_diff_config_path, imported_cluster_model_path],
        show_api=False,
    )
    model_path.change(
        uploaded_bundle_option_refresh_fn,
        [model_path],
        [enable_diffusion, enable_cluster],
        show_api=False,
    )
    workspace_model_selection.change(
        lambda selection: persist_selected_model("workspace", selection),
        [workspace_model_selection],
        [],
        show_api=False,
    )
    workspace_model_selection.change(
        model_checkpoint_refresh_fn,
        [workspace_model_selection],
        [workspace_model_checkpoint_selection],
        show_api=False,
    )
    workspace_model_selection.change(
        model_option_refresh_fn,
        [workspace_model_selection],
        [enable_diffusion, enable_cluster],
        show_api=False,
    )
    workspace_model_selection.change(
        model_extra_refresh_fn,
        [workspace_model_selection],
        [diff_model_path, diff_config_path, cluster_model_path],
        show_api=False,
    )
    imported_model_selection.change(
        lambda selection: persist_selected_model("imported", selection),
        [imported_model_selection],
        [],
        show_api=False,
    )
    imported_model_selection.change(
        model_checkpoint_refresh_fn,
        [imported_model_selection],
        [imported_model_checkpoint_selection],
        show_api=False,
    )
    imported_model_selection.change(
        model_option_refresh_fn,
        [imported_model_selection],
        [enable_diffusion, enable_cluster],
        show_api=False,
    )
    imported_model_selection.change(
        model_extra_refresh_fn,
        [imported_model_selection],
        [imported_diff_model_path, imported_diff_config_path, imported_cluster_model_path],
        show_api=False,
    )
    workspace_export_bundle_btn.click(
        export_local_model_bundle_action,
        [workspace_model_selection, workspace_model_checkpoint_selection],
        [workspace_export_bundle_status, workspace_export_bundle_path, workspace_open_export_location_btn],
        show_api=False,
    )
    workspace_open_export_location_btn.click(
        open_exported_file_location,
        [workspace_export_bundle_path],
        [],
        show_api=False,
    )
    imported_export_bundle_btn.click(
        export_local_model_bundle_action,
        [imported_model_selection, imported_model_checkpoint_selection],
        [imported_export_bundle_status, imported_export_bundle_path, imported_open_export_location_btn],
        show_api=False,
    )
    imported_open_export_location_btn.click(
        open_exported_file_location,
        [imported_export_bundle_path],
        [],
        show_api=False,
    )
    model_load_button.click(
        modelAnalysis,
        [model_path, config_path, cluster_model_path, device, enhance, diff_model_path, diff_config_path, only_diffusion, local_model_enabled, model_source, workspace_model_selection, workspace_model_checkpoint_selection, imported_model_selection, imported_model_checkpoint_selection, enable_diffusion, enable_cluster],
        [sid, sid_output],
        show_api=False,
    )
    model_unload_button.click(modelUnload, [], [sid, sid_output], show_api=False)
    vc_input3.change(describe_input_audio, [vc_input3], [input_audio_info], show_api=False)
    vc_input3.change(preview_input_audio, [vc_input3], [vc_input_preview], show_api=False)
    vc_submit.click(quality_vc_fn, [sid, vc_input3, quality_mode, vc_transform, cluster_ratio, k_step], [vc_output1, vc_output2], show_api=False)
    quality_mode.change(apply_quality_mode, [quality_mode], [cluster_ratio, k_step], show_api=False)
    best_quality_preset_btn.click(
        lambda: ("极致质量", 0, QUALITY_MODES["极致质量"]["cluster_ratio"], QUALITY_MODES["极致质量"]["k_step"]),
        [],
        [quality_mode, vc_transform, cluster_ratio, k_step],
        show_api=False,
    )
    export_summary_btn.click(
        export_runtime_summary,
        [sid, quality_mode, vc_transform, cluster_ratio, k_step],
        [runtime_summary_output],
        show_api=False,
    )
    server_port = find_available_port(7860)

    if os.environ.get("OPEN_BROWSER", "1") != "0":
        try:
            webbrowser.open(f"http://127.0.0.1:{server_port}")
        except Exception:
            pass
    app.launch(server_port=server_port)


 
