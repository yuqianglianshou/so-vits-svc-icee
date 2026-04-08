import logging
import os
import socket
import traceback
import webbrowser
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch

from gradio_api_info_fallback import apply_gradio_4_api_info_patch
from inference.infer_tool import Svc
from infer_ui.local_models import (
    LOCAL_MODEL_ROOT,
    load_last_selected_infer_model,
    local_model_refresh_fn,
    persist_local_model_selection,
    save_last_selected_infer_model,
    scan_local_models,
)
from infer_ui.convert import quality_convert
from infer_ui.files import resolve_model_inputs
from infer_ui.runtime import (
    build_loaded_model_result,
    export_runtime_summary_for_model,
    resolve_device_choice,
)
from infer_ui.text import (
    render_convert_result_html,
    render_load_result_html,
    render_readiness_html,
)
from quality_presets import BEST_QUALITY_PRESET, QUALITY_MODES

apply_gradio_4_api_info_patch()

ROOT = Path(__file__).resolve().parent
INFER_PAGE_CSS = (ROOT / "infer_ui" / "page.css").read_text(encoding="utf-8")


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

def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,local_model_enabled,local_model_selection):
    """统一处理上传模式和本地模式下的模型加载入口。"""
    try:
        if local_model_enabled:
            save_last_selected_infer_model(local_model_selection)
        model_path, config_path, cluster_model_path, diff_model_path, diff_config_path = resolve_model_inputs(
            model_path,
            config_path,
            cluster_model_path,
            diff_model_path,
            diff_config_path,
            local_model_enabled,
            local_model_selection,
        )
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
    initial_local_model_choices = scan_local_models()
    remembered_local_model = load_last_selected_infer_model()
    initial_local_model_selection = (
        remembered_local_model
        if remembered_local_model in initial_local_model_choices
        else (initial_local_model_choices[0] if initial_local_model_choices else None)
    )
    initial_local_model_enabled = bool(initial_local_model_selection)

    gr.HTML("""
        <div class="hero">
            <h2>高音质唱歌转换</h2>
        </div>
    """)

    local_model_enabled = gr.Checkbox(value=initial_local_model_enabled, visible=False)
    enhance = gr.Checkbox(value=False, visible=False)
    only_diffusion = gr.Checkbox(value=False, visible=False)
    with gr.Row():
        with gr.Column(scale=6, elem_classes=["step-card", "card-model"]):
            gr.Markdown("### 1. 模型文件")
            gr.Markdown("先选主模型和配置；推荐使用增强文件。")
            with gr.Tabs():
                with gr.TabItem('本地模型') as local_model_tab_local:
                    gr.Markdown("#### 必需")
                    gr.Markdown(f"本地模型目录：`{LOCAL_MODEL_ROOT}`")
                    local_model_refresh_btn = gr.Button('刷新本地模型列表', elem_classes=["info-action"], elem_id="infer-local-refresh")
                    local_model_selection = gr.Dropdown(
                        label='选择本地模型文件夹',
                        choices=initial_local_model_choices,
                        value=initial_local_model_selection,
                        interactive=True,
                    )
                    with gr.Accordion("可选增强文件", open=True):
                        with gr.Row():
                            diff_model_path = gr.File(label="音质增强模型 `.pt`")
                            diff_config_path = gr.File(label="音质增强配置 `.yaml`")
                        cluster_model_path = gr.File(label="音色增强文件 `.pkl` 或聚类文件")
                with gr.TabItem('手动上传') as local_model_tab_upload:
                    gr.Markdown("#### 必需")
                    with gr.Row():
                        model_path = gr.File(label="So-VITS 模型 `.pth`")
                        config_path = gr.File(label="模型配置 `.json`")
                    with gr.Accordion("可选增强文件", open=True):
                        with gr.Row():
                            diff_model_path = gr.File(label="音质增强模型 `.pt`")
                            diff_config_path = gr.File(label="音质增强配置 `.yaml`")
                        cluster_model_path = gr.File(label="音色增强文件 `.pkl` 或聚类文件")
        with gr.Column(scale=5, elem_classes=["step-card", "card-status"]):
            gr.Markdown("### 2. 加载与确认")
            device = gr.Dropdown(label="推理设备", choices=["Auto", *cuda.keys(), "cpu"], value="Auto")
            gr.Markdown("#### 准备情况")
            readiness_output = gr.HTML(
                render_readiness_html(None, None, None, None, None, initial_local_model_enabled, initial_local_model_selection)
            )
            model_load_button = gr.Button(value="加载模型", variant="primary", elem_classes=["primary-action"])
            model_unload_button = gr.Button(value="卸载模型", elem_classes=["danger-secondary"], elem_id="infer-unload")
            sid = gr.Dropdown(label="当前模型音色")
            gr.Markdown("#### 加载结果")
            sid_output = gr.HTML(render_load_result_html("先加载主模型和配置。"))

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

    local_model_refresh_btn.click(local_model_refresh_fn, outputs=local_model_selection, show_api=False)
    local_model_tab_upload.select(lambda: False, outputs=local_model_enabled, show_api=False)
    local_model_tab_local.select(lambda: True, outputs=local_model_enabled, show_api=False)
    model_path.change(
        render_readiness_html,
        [model_path, config_path, diff_model_path, diff_config_path, cluster_model_path, local_model_enabled, local_model_selection],
        [readiness_output],
        show_api=False,
    )
    config_path.change(
        render_readiness_html,
        [model_path, config_path, diff_model_path, diff_config_path, cluster_model_path, local_model_enabled, local_model_selection],
        [readiness_output],
        show_api=False,
    )
    diff_model_path.change(
        render_readiness_html,
        [model_path, config_path, diff_model_path, diff_config_path, cluster_model_path, local_model_enabled, local_model_selection],
        [readiness_output],
        show_api=False,
    )
    diff_config_path.change(
        render_readiness_html,
        [model_path, config_path, diff_model_path, diff_config_path, cluster_model_path, local_model_enabled, local_model_selection],
        [readiness_output],
        show_api=False,
    )
    cluster_model_path.change(
        render_readiness_html,
        [model_path, config_path, diff_model_path, diff_config_path, cluster_model_path, local_model_enabled, local_model_selection],
        [readiness_output],
        show_api=False,
    )
    local_model_selection.change(
        persist_local_model_selection,
        [local_model_selection],
        [],
        show_api=False,
    )
    local_model_selection.change(
        render_readiness_html,
        [model_path, config_path, diff_model_path, diff_config_path, cluster_model_path, local_model_enabled, local_model_selection],
        [readiness_output],
        show_api=False,
    )
    model_load_button.click(
        modelAnalysis,
        [model_path, config_path, cluster_model_path, device, enhance, diff_model_path, diff_config_path, only_diffusion, local_model_enabled, local_model_selection],
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


 
