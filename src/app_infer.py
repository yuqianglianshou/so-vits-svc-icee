import logging
import os
import socket
import webbrowser
from pathlib import Path

import gradio as gr
import torch

from src.gradio_api_info_fallback import apply_gradio_4_api_info_patch
from src.infer_ui.input_history import (
    build_input_audio_history_choices,
    handle_selected_history_audio,
    handle_uploaded_input_audio,
    refresh_input_audio_history,
)
from src.infer_ui.local_models import (
    IMPORTED_MODEL_ROOT,
    WORKSPACE_MODEL_ROOT,
    imported_model_refresh_fn,
    model_checkpoint_refresh_fn,
    model_diffusion_checkpoint_refresh_fn,
    model_extra_refresh_fn,
    model_option_refresh_fn,
    persist_selected_model,
    workspace_model_refresh_fn,
)
from src.infer_ui.model_state import build_initial_model_state
from src.infer_ui.model_actions import (
    export_local_model_bundle_action,
    open_exported_file_location,
    uploaded_bundle_option_refresh_fn,
)
from src.infer_ui.model_runtime import (
    export_runtime_summary as export_runtime_summary_for_current_model,
    safe_analyze_and_load_model,
    unload_model,
)
from src.infer_ui.convert import convert_tts_audio, quality_convert
from src.infer_ui.runtime import render_load_result_html
from src.infer_ui.text import (
    render_convert_result_html,
)
from src.quality_presets import BEST_QUALITY_PRESET, QUALITY_MODES

apply_gradio_4_api_info_patch()

CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent
INFER_PAGE_CSS = (CODE_ROOT / "infer_ui" / "page.css").read_text(encoding="utf-8")
INPUT_AUDIO_HISTORY_DIR = ROOT / "inference_data" / "inputs" / "history"


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

def modelAnalysis(model_path, config_path, cluster_model_path, device, enhance, diff_model_path, diff_config_path, only_diffusion, local_model_enabled, model_source, workspace_model_selection, workspace_model_checkpoint_selection, workspace_diffusion_checkpoint_selection, imported_model_selection, imported_model_checkpoint_selection, imported_diffusion_checkpoint_selection, enable_diffusion, enable_cluster):
    """统一处理上传模式和本地模式下的模型加载入口。"""
    global model
    model, sid_update, summary_html = safe_analyze_and_load_model(
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
        cuda_map=cuda,
        debug=debug,
    )
    return sid_update, summary_html


def export_runtime_summary(sid, quality_mode, vc_transform, cluster_ratio, k_step):
    """导出当前已加载模型的推理运行摘要。"""
    global model
    return export_runtime_summary_for_current_model(model, sid, quality_mode, vc_transform, cluster_ratio, k_step)

    
def modelUnload():
    """卸载当前模型，并清理显存缓存。"""
    global model
    model, sid_update, summary_html = unload_model(model)
    return sid_update, summary_html

def apply_quality_mode(mode):
    """根据质量模式同步推荐的检索比例和浅扩散步数。"""
    preset = QUALITY_MODES[mode]
    return preset["cluster_ratio"], preset["k_step"]


def quality_vc_fn(sid, input_audio, quality_mode, vc_transform, cluster_ratio, k_step):
    """推理页主入口：按当前高质量预设执行一次转换。"""
    global model
    return quality_convert(model, sid, input_audio, quality_mode, vc_transform, cluster_ratio, k_step, BEST_QUALITY_PRESET)


def tts_vc_fn(sid, text, lang, gender, rate, volume, quality_mode, vc_transform, cluster_ratio, k_step):
    """文本转语音后再走当前模型转换。"""
    global model
    return convert_tts_audio(
        model,
        text,
        lang,
        gender,
        rate,
        volume,
        sid,
        vc_transform,
        BEST_QUALITY_PRESET["auto_predict_f0"],
        cluster_ratio,
        BEST_QUALITY_PRESET["slice_db"],
        BEST_QUALITY_PRESET["noise_scale"],
        BEST_QUALITY_PRESET["pad_seconds"],
        BEST_QUALITY_PRESET["clip_seconds"],
        BEST_QUALITY_PRESET["linear_gradient"],
        BEST_QUALITY_PRESET["linear_gradient_retain"],
        BEST_QUALITY_PRESET["f0_predictor"],
        BEST_QUALITY_PRESET["enhancer_adaptive_key"],
        BEST_QUALITY_PRESET["cr_threshold"],
        k_step,
        BEST_QUALITY_PRESET["second_encoding"],
        BEST_QUALITY_PRESET["loudness_envelope_adjustment"],
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
    initial_state = build_initial_model_state()
    initial_input_audio_history = build_input_audio_history_choices(INPUT_AUDIO_HISTORY_DIR)

    gr.HTML("""
        <div class="hero">
            <h2>高音质唱歌转换</h2>
        </div>
    """)

    local_model_enabled = gr.Checkbox(value=initial_state.local_model_enabled, visible=False)
    model_source = gr.State(value=initial_state.model_source)
    enhance = gr.Checkbox(value=False, visible=False)
    only_diffusion = gr.Checkbox(value=False, visible=False)
    with gr.Row():
        with gr.Column(scale=6, elem_classes=["step-card", "card-model"]):
            gr.Markdown("### 1. 模型文件")
            gr.Markdown("选择训练工作区、已导入模型，或上传单文件包。")
            with gr.Tabs():
                with gr.TabItem('训练工作区') as workspace_model_tab:
                    gr.Markdown(f"训练工作区目录：`{WORKSPACE_MODEL_ROOT}`")
                    workspace_model_refresh_btn = gr.Button('刷新训练工作区列表', elem_classes=["info-action"], elem_id="infer-workspace-refresh")
                    workspace_model_selection = gr.Dropdown(
                        label='选择训练工作区模型',
                        choices=initial_state.workspace_choices,
                        value=initial_state.workspace_selection,
                        interactive=True,
                    )
                    workspace_model_checkpoint_selection = gr.Dropdown(
                        label='选择主模型 G_*.pth',
                        choices=initial_state.workspace_checkpoints,
                        value=initial_state.workspace_checkpoint_selection,
                        interactive=bool(initial_state.workspace_checkpoints),
                    )
                    with gr.Accordion("音质增强和音色增强文件", open=True):
                        with gr.Row():
                            diff_model_path = gr.Dropdown(label="选择音质增强模型 `.pt`", choices=initial_state.workspace_diffusion_checkpoints, value=initial_state.workspace_diffusion_checkpoint_selection, interactive=bool(initial_state.workspace_diffusion_checkpoints))
                            diff_config_path = gr.Textbox(label="音质增强配置 `.yaml`", value=initial_state.workspace_diff_config_path or "", interactive=False)
                        cluster_model_path = gr.Textbox(label="音色增强索引 `.pkl`", value=initial_state.workspace_cluster_model_path or "", interactive=False)
                    workspace_export_bundle_status = gr.HTML('<div class="subtle-note">会基于当前选中的 G_*.pth 导出当前模型包；若目录里存在扩散、索引，也会自动一起打包。</div>')
                    workspace_export_bundle_path = gr.Textbox(label="生成路径", value="", interactive=False)
                    with gr.Row():
                        workspace_export_bundle_btn = gr.Button("导出当前模型包", elem_classes=["info-action"])
                        workspace_open_export_location_btn = gr.Button("打开文件位置", interactive=False, elem_classes=["info-action"])
                with gr.TabItem('已导入的模型') as imported_model_tab:
                    gr.Markdown(f"已导入模型目录：`{IMPORTED_MODEL_ROOT}`")
                    imported_model_refresh_btn = gr.Button('刷新已导入模型列表', elem_classes=["info-action"], elem_id="infer-imported-refresh")
                    imported_model_selection = gr.Dropdown(
                        label='选择已导入的模型',
                        choices=initial_state.imported_choices,
                        value=initial_state.imported_selection,
                        interactive=True,
                    )
                    imported_model_checkpoint_selection = gr.Dropdown(
                        label='选择主模型 G_*.pth',
                        choices=initial_state.imported_checkpoints,
                        value=initial_state.imported_checkpoint_selection,
                        interactive=bool(initial_state.imported_checkpoints),
                    )
                    with gr.Accordion("音质增强和音色增强文件", open=True):
                        with gr.Row():
                            imported_diff_model_path = gr.Dropdown(label="选择音质增强模型 `.pt`", choices=initial_state.imported_diffusion_checkpoints, value=initial_state.imported_diffusion_checkpoint_selection, interactive=bool(initial_state.imported_diffusion_checkpoints))
                            imported_diff_config_path = gr.Textbox(label="音质增强配置 `.yaml`", value=initial_state.imported_diff_config_path or "", interactive=False)
                        imported_cluster_model_path = gr.Textbox(label="音色增强索引 `.pkl`", value=initial_state.imported_cluster_model_path or "", interactive=False)
                    imported_export_bundle_status = gr.HTML('<div class="subtle-note">会基于当前选中的 G_*.pth 导出当前模型包；若目录里存在扩散、索引，也会自动一起打包。</div>')
                    imported_export_bundle_path = gr.Textbox(label="生成路径", value="", interactive=False)
                    with gr.Row():
                        imported_export_bundle_btn = gr.Button("导出当前模型包", elem_classes=["info-action"])
                        imported_open_export_location_btn = gr.Button("打开文件位置", interactive=False, elem_classes=["info-action"])
                with gr.TabItem('手动上传') as local_model_tab_upload:
                    model_path = gr.File(label="单文件推理包 `.pth`")
                    config_path = gr.State(value=None)
        with gr.Column(scale=5, elem_classes=["step-card", "card-status"]):
            gr.Markdown("### 2. 加载与确认")
            device = gr.Dropdown(label="推理设备", choices=["Auto", *cuda.keys(), "cpu"], value="Auto")
            with gr.Row():
                enable_diffusion = gr.Checkbox(label="音质增强", value=initial_state.has_diffusion, interactive=initial_state.has_diffusion)
                enable_cluster = gr.Checkbox(label="音色增强", value=initial_state.has_cluster, interactive=initial_state.has_cluster)
            with gr.Row():
                model_load_button = gr.Button(value="加载模型", variant="primary", elem_classes=["primary-action"])
                model_unload_button = gr.Button(value="卸载模型", elem_classes=["danger-secondary"], elem_id="infer-unload")
            sid = gr.Dropdown(label="当前模型音色")
            gr.Markdown("#### 加载结果")
            sid_output = gr.HTML(render_load_result_html("先加载主模型包。"))

    with gr.Row():
        with gr.Column(scale=5, elem_classes=["step-card", "card-params"]):
            gr.Markdown("### 3. 高音质参数")
            gr.Markdown("通常只需选质量模式，再改 `变调`。")
            quality_mode = gr.Radio(
                label="质量模式",
                choices=["标准高质", "极致质量"],
                value="极致质量",
            )
            vc_transform = gr.Number(label="变调（半音）", value=0)
            cluster_ratio = gr.Slider(label="特征检索混合比例", minimum=0, maximum=1, step=0.05, value=QUALITY_MODES["极致质量"]["cluster_ratio"])
            k_step = gr.Slider(label="浅扩散步数", minimum=1, maximum=1000, step=1, value=QUALITY_MODES["极致质量"]["k_step"])
            with gr.Row():
                best_quality_preset_btn = gr.Button(value="恢复推荐参数", elem_classes=["success-action"], elem_id="infer-preset")
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
                - 输出会同时提供 `flac / wav` 下载
                - F0 预测固定为 `rmvpe`
                - 其余参数已按高质量预设固定
                """)
        with gr.Column(scale=6, elem_classes=["step-card", "card-output"]):
            gr.Markdown("### 4. 上传音频并转换")
            with gr.Row():
                input_history_selection = gr.Dropdown(
                    label="之前上传过的音频",
                    choices=initial_input_audio_history,
                    value=None,
                    interactive=bool(initial_input_audio_history),
                )
                input_history_refresh_btn = gr.Button("刷新列表", elem_classes=["info-action"])
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
            vc_output2 = gr.Audio(label="输出音频试听", interactive=False, elem_classes=["result-audio"])
            with gr.Row():
                vc_output3 = gr.File(label="下载 `.flac`", interactive=False)
                vc_output4 = gr.File(label="下载 `.wav`", interactive=False)

    with gr.Row():
        with gr.Column(scale=12, elem_classes=["step-card", "card-output"]):
            gr.Markdown("### 5. 文本转语音并转换")
            gr.Markdown("先用 EdgeTTS 生成语音，再走当前模型转换。")
            tts_text = gr.Textbox(label="输入文本", lines=3, placeholder="输入想要合成并转换的文本")
            with gr.Row():
                tts_lang = gr.Dropdown(
                    label="语音语言",
                    choices=["Auto", "zh-CN", "zh-TW", "en-US", "ja-JP", "ko-KR"],
                    value="Auto",
                )
                tts_gender = gr.Dropdown(
                    label="音色性别",
                    choices=["女", "男"],
                    value="女",
                )
            with gr.Row():
                tts_rate = gr.Slider(label="语速", minimum=-1.0, maximum=1.0, step=0.05, value=0.0)
                tts_volume = gr.Slider(label="音量", minimum=-1.0, maximum=1.0, step=0.05, value=0.0)
                tts_submit = gr.Button("开始文本转语音并转换", variant="primary", elem_classes=["primary-action"])
            gr.Markdown("#### 处理结果")
            tts_output1 = gr.HTML(render_convert_result_html("等待开始转换。"))
            tts_output2 = gr.Audio(label="输出音频试听", interactive=False, elem_classes=["result-audio"])
            with gr.Row():
                tts_output3 = gr.File(label="下载 `.flac`", interactive=False)
                tts_output4 = gr.File(label="下载 `.wav`", interactive=False)

    workspace_model_refresh_btn.click(workspace_model_refresh_fn, outputs=workspace_model_selection, show_api=False).then(
        model_checkpoint_refresh_fn,
        [workspace_model_selection],
        [workspace_model_checkpoint_selection],
        show_api=False,
    ).then(
        model_diffusion_checkpoint_refresh_fn,
        [workspace_model_selection],
        [diff_model_path],
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
        model_diffusion_checkpoint_refresh_fn,
        [imported_model_selection],
        [imported_diff_model_path],
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
    local_model_tab_upload.select(lambda: "upload", outputs=model_source, show_api=False)
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
        model_diffusion_checkpoint_refresh_fn,
        [workspace_model_selection],
        [diff_model_path],
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
        model_diffusion_checkpoint_refresh_fn,
        [imported_model_selection],
        [imported_diff_model_path],
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
        [model_path, config_path, cluster_model_path, device, enhance, diff_model_path, diff_config_path, only_diffusion, local_model_enabled, model_source, workspace_model_selection, workspace_model_checkpoint_selection, diff_model_path, imported_model_selection, imported_model_checkpoint_selection, imported_diff_model_path, enable_diffusion, enable_cluster],
        [sid, sid_output],
        show_api=False,
    )
    model_unload_button.click(modelUnload, [], [sid, sid_output], show_api=False)
    input_history_refresh_btn.click(lambda: refresh_input_audio_history(INPUT_AUDIO_HISTORY_DIR), [], [input_history_selection], show_api=False)
    vc_input3.change(lambda audio_path: handle_uploaded_input_audio(INPUT_AUDIO_HISTORY_DIR, audio_path), [vc_input3], [input_history_selection, input_audio_info, vc_input_preview], show_api=False)
    input_history_selection.change(handle_selected_history_audio, [input_history_selection], [input_audio_info, vc_input_preview], show_api=False)
    vc_submit.click(quality_vc_fn, [sid, input_history_selection, quality_mode, vc_transform, cluster_ratio, k_step], [vc_output1, vc_output2, vc_output3, vc_output4], show_api=False)
    tts_submit.click(
        tts_vc_fn,
        [sid, tts_text, tts_lang, tts_gender, tts_rate, tts_volume, quality_mode, vc_transform, cluster_ratio, k_step],
        [tts_output1, tts_output2, tts_output3, tts_output4],
        show_api=False,
    )
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


 
