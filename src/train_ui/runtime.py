from __future__ import annotations

import time
from pathlib import Path

from src.train_ui.panels import build_preflight_check_html, build_runtime_banner_text
from src.train_ui.paths import ROOT, resolve_raw_dataset_dir, sanitize_model_name
from src.train_ui.state import collect_stage_state
from src.train_ui.text import format_duration, format_duration_clock
from src.train_ui.workspace import count_raw_dataset_wavs, count_training_wavs


def resolve_task_log_path(active_task: dict) -> Path | None:
    return active_task.get("display_log_path") or active_task.get("log_path")


def detect_cuda_status() -> str:
    try:
        import torch
    except Exception as exc:
        return f"PyTorch 不可用：{exc}"
    if not torch.cuda.is_available():
        return "CUDA 不可用"
    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        device_name = "已检测到 CUDA 设备"
    return f"CUDA 可用：{device_name}"


def render_preflight_check(
    model_name: str,
    raw_dir: str,
    train_dir: str,
    *,
    get_sovits_g0_path_fn,
    get_sovits_d0_path_fn,
    get_diffusion_model_0_path_fn,
    get_rmvpe_path_fn,
    is_rmvpe_asset_valid_fn,
    get_contentvec_hf_path_fn,
    get_nsf_hifigan_model_path_fn,
    get_nsf_hifigan_config_path_fn,
) -> str:
    model_name = sanitize_model_name(model_name)
    stage_state = collect_stage_state(model_name, raw_dir, train_dir)
    raw_relative_dir = resolve_raw_dataset_dir(raw_dir)
    _, raw_wavs = count_raw_dataset_wavs(ROOT / raw_relative_dir)
    train_speakers, train_wavs = count_training_wavs(ROOT / train_dir)
    cuda_status = detect_cuda_status()

    train_requirements: list[str] = []
    if "CUDA 可用" not in cuda_status:
        train_requirements.append("主模型训练和扩散训练需要 Windows/Linux + NVIDIA GPU + CUDA 环境。")
    if not get_sovits_g0_path_fn().exists():
        train_requirements.append(f"缺少 So-VITS 生成器底模：{get_sovits_g0_path_fn().relative_to(ROOT).as_posix()}")
    if not get_sovits_d0_path_fn().exists():
        train_requirements.append(f"缺少 So-VITS 判别器底模：{get_sovits_d0_path_fn().relative_to(ROOT).as_posix()}")
    if not get_diffusion_model_0_path_fn().exists():
        train_requirements.append(f"缺少扩散底模：{get_diffusion_model_0_path_fn().relative_to(ROOT).as_posix()}")
    if not get_rmvpe_path_fn().exists():
        train_requirements.append(f"缺少 RMVPE 预训练文件：{get_rmvpe_path_fn().relative_to(ROOT).as_posix()}")
    elif not is_rmvpe_asset_valid_fn():
        train_requirements.append(f"RMVPE 预训练文件已损坏，请重新导入：{get_rmvpe_path_fn().relative_to(ROOT).as_posix()}")
    contentvec_hf_dir = get_contentvec_hf_path_fn()
    if not ((contentvec_hf_dir / "config.json").exists() and (contentvec_hf_dir / "model.safetensors").exists()):
        train_requirements.append(f"缺少 ContentVec HF 模型目录：{contentvec_hf_dir.relative_to(ROOT).as_posix()}/")
    if not (get_nsf_hifigan_model_path_fn().exists() and get_nsf_hifigan_config_path_fn().exists()):
        train_requirements.append(f"缺少 NSF-HIFIGAN 声码器：{get_nsf_hifigan_model_path_fn().parent.relative_to(ROOT).as_posix()}/")
    if raw_wavs == 0:
        train_requirements.append(f"{raw_relative_dir.as_posix()} 为空，无法开始完整训练流程。")
    if train_wavs > 0 and train_speakers != 1:
        train_requirements.append("处理后数据目录应直接包含当前数据集的 wav 文件；当前目录结构不符合最新规则。")

    summary = stage_state["summary"]
    for needle, message in (
        ("3. 提取特征：等待上一步", "特征预处理还不能开始，先补齐配置与文件列表。"),
        ("4. 主模型训练：等待上一步", "主模型训练还不能开始，先完成特征预处理。"),
        ("5. 扩散训练：等待上一步", "扩散训练还不能开始，先完成特征预处理。"),
        ("6. 训练音色增强索引：等待上一步", "音色增强索引还不能开始，先完成特征预处理。"),
    ):
        if needle in summary:
            train_requirements.append(message)

    head = [
        (
            f"当前模型：{model_name}；当前模型数据目录：{raw_relative_dir.as_posix()}",
            "#1f8f4c" if raw_wavs > 0 else "#d97706",
        ),
        (
            f"环境：{cuda_status}",
            "#1f8f4c" if "CUDA 可用" in cuda_status else "#c0392b",
        ),
        (
            f"当前数据：{raw_relative_dir.as_posix()}，{raw_wavs} 个 wav",
            "#1f8f4c" if raw_wavs > 0 else "#c0392b",
        ),
    ]

    info_rows = [
        '<div class="stage-check-row">'
        f'<div class="stage-check-title"><span class="stage-dot" style="color:{color};">●</span>{line}</div>'
        "</div>"
        for line, color in head
    ]
    return build_preflight_check_html(info_rows, train_requirements)


def task_runtime_text(active_task: dict, *, current_stage_label_fn, pipeline_labels: dict) -> str:
    proc = active_task["proc"]
    thread = active_task["thread"]
    pipeline_name = active_task["pipeline_name"]
    if proc is None and not (thread is not None and thread.is_alive()):
        return "当前没有运行中的任务。"
    started = active_task["started_at"] or time.time()
    elapsed = int(time.time() - started)
    elapsed_text = format_duration(elapsed)
    if proc is None and thread is not None and thread.is_alive():
        status = "运行中（流程编排中）"
    else:
        status = "运行中" if proc.poll() is None else f"已结束（退出码 {proc.returncode}）"
    task_display = pipeline_labels.get(pipeline_name, pipeline_name) if pipeline_name else active_task["name"]
    log_path = resolve_task_log_path(active_task)
    return (
        f"任务：{task_display}\n"
        f"阶段：{current_stage_label_fn()}\n"
        f"状态：{status}\n"
        f"已运行：{elapsed_text}\n"
        f"日志：{log_path}\n"
        f"命令：{' '.join(active_task['cmd']) if active_task['cmd'] else '等待当前子步骤启动'}"
    )


def render_runtime_banner(active_task: dict, *, current_stage_label_fn, pipeline_labels: dict) -> str:
    proc = active_task["proc"]
    thread = active_task["thread"]
    pipeline_name = active_task["pipeline_name"]
    started = active_task["started_at"]
    is_running = (proc is not None and proc.poll() is None) or (thread is not None and thread.is_alive())

    task_display = pipeline_labels.get(pipeline_name, pipeline_name) if pipeline_name else current_stage_label_fn()
    elapsed_text = format_duration_clock(max(0, int(time.time() - started))) if started is not None else ""
    exit_code = proc.returncode if proc is not None else None
    return build_runtime_banner_text(started, task_display, is_running, elapsed_text, exit_code)


def tail_log(path: Path | None, max_lines: int = 80) -> str:
    if path is None or not path.exists():
        return "日志文件尚不存在。"
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:]) if lines else "日志暂时为空。"


def render_task_panel_snapshot(
    model_name: str,
    raw_dir: str,
    train_dir: str,
    *,
    active_task: dict,
    render_stage_alert_fn,
    current_task_feedback_fn,
    current_stage_label_fn,
    pipeline_labels: dict,
):
    log_path = resolve_task_log_path(active_task)
    return (
        render_stage_alert_fn(model_name, raw_dir, train_dir),
        render_runtime_banner(active_task, current_stage_label_fn=current_stage_label_fn, pipeline_labels=pipeline_labels),
        current_task_feedback_fn(model_name, raw_dir, train_dir),
        task_runtime_text(active_task, current_stage_label_fn=current_stage_label_fn, pipeline_labels=pipeline_labels),
        tail_log(log_path),
    )
