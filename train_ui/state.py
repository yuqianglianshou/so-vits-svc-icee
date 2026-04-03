"""训练页阶段状态判断与阶段展示。

这个模块专门负责：
1. 根据当前模型和数据目录判断 1-6 步所处阶段，
2. 生成训练前检查区需要的阶段列表，
3. 生成按钮状态所需的基础数据。

这样训练页入口可以把精力放在事件绑定和任务执行上。
"""

from pathlib import Path

from train_ui.paths import (
    ROOT,
    model_config_path,
    model_diff_config_path,
    model_diffusion_dir,
    model_index_path,
    model_root_dir,
    model_train_list_path,
    model_val_list_path,
    resolve_raw_dataset_dir,
    sanitize_model_name,
)
from train_ui.text import compact_stage_line, status_tone_tag
from train_ui.workspace import count_raw_dataset_wavs, count_training_wavs, speaker_dirs_in_train_root


def count_matching_files(base_dir: Path, pattern: str):
    """统计目录下匹配模式的文件数量。"""
    if not base_dir.exists():
        return 0
    return len(list(base_dir.rglob(pattern)))


def count_generated_checkpoints(base_dir: Path, pattern: str, excluded_names: set[str]):
    """统计训练真正产出的检查点文件，排除预训练底模。"""
    if not base_dir.exists():
        return 0
    return sum(1 for path in base_dir.glob(pattern) if path.name not in excluded_names)


def count_nonempty_lines(path: Path):
    """统计文本文件中的非空行数量。"""
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def collect_stage_state(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "dataset/44k"):
    """汇总训练阶段状态，供进度区和按钮状态统一使用。"""
    model_name = sanitize_model_name(model_name)
    raw_relative_dir = resolve_raw_dataset_dir(raw_dir)
    raw_root = ROOT / raw_relative_dir
    train_root = ROOT / train_dir
    main_model_dir = model_root_dir(model_name)
    diff_model_dir = model_diffusion_dir(model_name)
    index_file = model_index_path(model_name)

    raw_speakers, raw_wavs = count_raw_dataset_wavs(raw_root)
    train_speakers, train_wavs = count_training_wavs(train_root)
    train_speaker_dirs = speaker_dirs_in_train_root(train_root)
    soft_count = count_matching_files(train_root, "*.soft.pt")
    f0_count = count_matching_files(train_root, "*.f0.npy")
    spec_count = count_matching_files(train_root, "*.spec.pt")
    train_bundle_count = count_matching_files(train_root, "*.train.pt")
    main_ckpt_count = count_generated_checkpoints(main_model_dir, "G_*.pth", {"G_0.pth"})
    diff_ckpt_count = count_generated_checkpoints(diff_model_dir, "model_*.pt", {"model_0.pt"})

    has_config = model_config_path(model_name).exists()
    has_diff_config = model_diff_config_path(model_name).exists()
    train_list_path = model_train_list_path(model_name)
    val_list_path = model_val_list_path(model_name)
    has_train_list = train_list_path.exists()
    has_val_list = val_list_path.exists()
    train_list_entries = count_nonempty_lines(train_list_path)
    val_list_entries = count_nonempty_lines(val_list_path)
    single_speaker_ready = train_speakers == 1
    feature_progress_detail = (
        f"wav={train_wavs} / soft={soft_count} / f0={f0_count} / spec={spec_count} / trainpt={train_bundle_count}"
    )

    stage_lines = []
    stage_items = []
    button_state = {}

    if raw_wavs == 0:
        stage_lines.append(compact_stage_line("1. 重采样", "未满足前置条件", "缺少原始 wav"))
        stage_items.append(("1. 重采样", "未满足前置条件", "缺少原始 wav"))
        next_step = f"先准备 {raw_relative_dir.as_posix()} 数据集。"
        button_state["resample"] = {"value": f"1. 重采样到 {train_dir}", "interactive": False}
    elif train_wavs > 0:
        stage_lines.append(compact_stage_line("1. 重采样", "已完成", f"{train_wavs} 个 wav"))
        stage_items.append(("1. 重采样", "已完成", f"{train_wavs} 个 wav"))
        next_step = None
        button_state["resample"] = {"value": f"1. 重采样到 {train_dir}（已完成）", "interactive": True}
    else:
        stage_lines.append(compact_stage_line("1. 重采样", "可执行", f"{raw_wavs} 个 wav，直接开始"))
        stage_items.append(("1. 重采样", "可执行", f"{raw_wavs} 个 wav，直接开始"))
        next_step = f"执行第 1 步：重采样到 {train_dir}。"
        button_state["resample"] = {"value": f"1. 重采样到 {train_dir}", "interactive": True}

    config_ready = (
        has_config
        and has_diff_config
        and has_train_list
        and has_val_list
        and train_list_entries > 0
    )
    if train_wavs == 0:
        stage_lines.append(compact_stage_line("2. 生成配置与文件列表", "等待上一步", "等待重采样"))
        stage_items.append(("2. 生成配置与文件列表", "等待上一步", "等待重采样"))
        button_state["config"] = {"value": "2. 生成配置与文件列表（等待前置）", "interactive": False}
    elif not single_speaker_ready:
        detail = f"处理后数据目录中只能保留 1 个训练数据子目录，当前检测到 {train_speakers} 个：{'、'.join(train_speaker_dirs)}"
        stage_lines.append(compact_stage_line("2. 生成配置与文件列表", "未满足前置条件", detail))
        stage_items.append(("2. 生成配置与文件列表", "未满足前置条件", detail))
        if next_step is None:
            next_step = "先清理处理后数据目录，确保本轮训练只保留 1 个训练数据子目录。"
        button_state["config"] = {"value": "2. 生成配置与文件列表（等待数据目录整理）", "interactive": False}
    elif config_ready:
        detail = f"配置和列表已就绪（train={train_list_entries} / val={val_list_entries}）"
        stage_lines.append(compact_stage_line("2. 生成配置与文件列表", "已完成", detail))
        stage_items.append(("2. 生成配置与文件列表", "已完成", detail))
        button_state["config"] = {"value": "2. 生成配置与文件列表（已完成）", "interactive": True}
    else:
        missing = []
        if not has_config:
            missing.append(model_config_path(model_name).relative_to(ROOT).as_posix())
        if not has_diff_config:
            missing.append(model_diff_config_path(model_name).relative_to(ROOT).as_posix())
        if not has_train_list:
            missing.append(model_train_list_path(model_name).relative_to(ROOT).as_posix())
        if not has_val_list:
            missing.append(model_val_list_path(model_name).relative_to(ROOT).as_posix())
        if has_train_list and train_list_entries == 0:
            missing.append(f"{train_list_path.relative_to(ROOT).as_posix()}（内容为空）")
        stage_lines.append(compact_stage_line("2. 生成配置与文件列表", "可执行", "缺少：" + "、".join(missing)))
        stage_items.append(("2. 生成配置与文件列表", "可执行", "缺少：" + "、".join(missing)))
        if next_step is None:
            next_step = "执行第 2 步：生成配置与文件列表。"
        button_state["config"] = {"value": "2. 生成配置与文件列表", "interactive": True}

    feature_ready = (
        train_wavs > 0
        and soft_count >= train_wavs
        and f0_count >= train_wavs
        and spec_count >= train_wavs
        and train_bundle_count >= train_wavs
    )
    if not config_ready or not single_speaker_ready:
        stage_lines.append(compact_stage_line("3. 提取特征", "等待上一步", "等待配置与列表"))
        stage_items.append(("3. 提取特征", "等待上一步", "等待配置与列表"))
        button_state["preprocess"] = {"value": "3. 提取特征（等待前置）", "interactive": False}
    elif feature_ready:
        stage_lines.append(compact_stage_line("3. 提取特征", "已完成", feature_progress_detail))
        stage_items.append(("3. 提取特征", "已完成", feature_progress_detail))
        if next_step is None:
            next_step = f"执行第 4 步：启动主模型训练（{model_name}）。"
        button_state["preprocess"] = {"value": "3. 提取特征（已完成）", "interactive": True}
    else:
        stage_lines.append(compact_stage_line("3. 提取特征", "可执行", feature_progress_detail))
        stage_items.append(("3. 提取特征", "可执行", feature_progress_detail))
        next_step = f"第 3 步未完成：{feature_progress_detail}"
        button_state["preprocess"] = {"value": "3. 提取特征", "interactive": True}

    if not feature_ready or not single_speaker_ready:
        stage_lines.append(compact_stage_line("4. 主模型训练", "等待上一步", "等待特征完成"))
        stage_items.append(("4. 主模型训练", "等待上一步", "等待特征完成"))
        button_state["train"] = {"value": "4. 启动主模型训练（等待前置）", "interactive": False}
    elif main_ckpt_count > 0:
        stage_lines.append(compact_stage_line("4. 主模型训练", "已开始或已完成", f"已有 {main_ckpt_count} 个 G_*.pth"))
        stage_items.append(("4. 主模型训练", "已开始或已完成", f"已有 {main_ckpt_count} 个 G_*.pth"))
        if next_step is None:
            next_step = "如果主模型质量已稳定，可以进入第 5 步扩散训练。"
        button_state["train"] = {"value": "4. 启动主模型训练（已有产物）", "interactive": True}
    else:
        stage_lines.append(compact_stage_line("4. 主模型训练", "可执行", f"输出到 logs/{model_name}"))
        stage_items.append(("4. 主模型训练", "可执行", f"输出到 logs/{model_name}"))
        if next_step is None:
            next_step = f"执行第 4 步：启动主模型训练（{model_name}）。"
        button_state["train"] = {"value": "4. 启动主模型训练", "interactive": True}

    if not single_speaker_ready:
        stage_lines.append(compact_stage_line("5. 扩散训练", "等待上一步", "等待数据目录整理"))
        stage_items.append(("5. 扩散训练", "等待上一步", "等待数据目录整理"))
        button_state["train_diff"] = {"value": "5. 启动扩散训练（等待前置）", "interactive": False}
    elif main_ckpt_count == 0:
        stage_lines.append(compact_stage_line("5. 扩散训练", "等待上一步", "等待主模型"))
        stage_items.append(("5. 扩散训练", "等待上一步", "等待主模型"))
        button_state["train_diff"] = {"value": "5. 启动扩散训练（等待前置）", "interactive": False}
    elif diff_ckpt_count > 0:
        stage_lines.append(compact_stage_line("5. 扩散训练", "已开始或已完成", f"已有 {diff_ckpt_count} 个 model_*.pt"))
        stage_items.append(("5. 扩散训练", "已开始或已完成", f"已有 {diff_ckpt_count} 个 model_*.pt"))
        if next_step is None:
            next_step = "如果扩散模型也已满足听感，可以进入第 6 步训练音色增强索引。"
        button_state["train_diff"] = {"value": "5. 启动扩散训练（已有产物）", "interactive": True}
    else:
        stage_lines.append(compact_stage_line("5. 扩散训练", "可执行", f"输出到 logs/{model_name}/diffusion"))
        stage_items.append(("5. 扩散训练", "可执行", f"输出到 logs/{model_name}/diffusion"))
        if next_step is None:
            next_step = "执行第 5 步：启动扩散训练。"
        button_state["train_diff"] = {"value": "5. 启动扩散训练", "interactive": True}

    if not single_speaker_ready:
        stage_lines.append(compact_stage_line("6. 训练音色增强索引", "等待上一步", "等待数据目录整理"))
        stage_items.append(("6. 训练音色增强索引", "等待上一步", "等待数据目录整理"))
        button_state["train_index"] = {"value": "6. 训练音色增强索引（等待前置）", "interactive": False}
    elif main_ckpt_count == 0:
        stage_lines.append(compact_stage_line("6. 训练音色增强索引", "等待上一步", "等待主模型"))
        stage_items.append(("6. 训练音色增强索引", "等待上一步", "等待主模型"))
        button_state["train_index"] = {"value": "6. 训练音色增强索引（等待前置）", "interactive": False}
    elif index_file.exists():
        stage_lines.append(compact_stage_line("6. 训练音色增强索引", "已完成", "索引文件已生成"))
        stage_items.append(("6. 训练音色增强索引", "已完成", "索引文件已生成"))
        if next_step is None:
            next_step = "训练链路已基本齐备，可以转到推理界面做听感验收。"
        button_state["train_index"] = {"value": "6. 训练音色增强索引（已完成）", "interactive": True}
    else:
        stage_lines.append(compact_stage_line("6. 训练音色增强索引", "可执行", f"输出到 logs/{model_name}/feature_and_index.pkl"))
        stage_items.append(("6. 训练音色增强索引", "可执行", f"输出到 logs/{model_name}/feature_and_index.pkl"))
        if next_step is None:
            next_step = "执行第 6 步：训练音色增强索引。"
        button_state["train_index"] = {"value": "6. 训练音色增强索引", "interactive": True}

    if next_step is None:
        next_step = "当前没有更早的阻塞步骤，建议结合 TensorBoard 和听感结果决定是否继续训练。"

    summary = "\n\n".join(stage_lines) + f"\n\n下一步：{next_step}"
    return {
        "summary": summary,
        "stage_items": stage_items,
        "next_step": next_step,
        "button_state": button_state,
        "raw_wavs": raw_wavs,
        "raw_speakers": raw_speakers,
        "train_wavs": train_wavs,
        "train_speakers": train_speakers,
        "raw_relative_dir": raw_relative_dir,
        "feature_progress_detail": feature_progress_detail,
    }


def render_stage_judgement_html(stage_state: dict):
    """把阶段状态渲染成训练前检查区使用的 HTML。"""
    rows = []
    for title, status, detail in stage_state["stage_items"]:
        tone, dot = status_tone_tag(status)
        rows.append(
            '<div class="stage-check-row">'
            f'<div class="stage-check-title"><span class="stage-dot" style="color:{"#1f8f4c" if tone=="ok" else "#d97706" if tone=="warn" else "#c0392b"};">{dot}</span>{title}</div>'
            f'<div class="stage-check-detail">{detail}</div>'
            "</div>"
        )
    rows.append('<div class="stage-next-step">' f'下一步：<span>{stage_state["next_step"]}</span>' "</div>")
    return "".join(rows)


def build_button_state_maps(stage_state: dict, active_task_name, active_pipeline_name, is_running: bool, elapsed_text: str):
    """根据阶段状态和当前运行任务生成按钮展示状态。"""
    button_state = {key: value.copy() for key, value in stage_state["button_state"].items()}
    pipeline_button_state = {
        "pipeline_prep": {"value": "一键执行 1-3 步", "interactive": True},
        "pipeline_train": {"value": "一键执行到主模型训练", "interactive": True},
    }

    if is_running:
        for item in button_state.values():
            item["interactive"] = False
        for item in pipeline_button_state.values():
            item["interactive"] = False

        task_key_map = {
            "resample": "resample",
            "preprocess_flist_config": "config",
            "preprocess_hubert_f0": "preprocess",
            "train_main": "train",
            "train_diff": "train_diff",
            "train_index": "train_index",
        }
        active_task_key = task_key_map.get(active_task_name)
        if active_task_key in button_state:
            base_label = button_state[active_task_key]["value"].split("（")[0]
            button_state[active_task_key]["value"] = f"{base_label}（运行中 {elapsed_text}）"

        pipeline_key_map = {
            "pipeline_prep": "pipeline_prep",
            "pipeline_train_main": "pipeline_train",
        }
        active_pipeline_key = pipeline_key_map.get(active_pipeline_name)
        if active_pipeline_key in pipeline_button_state:
            base_label = pipeline_button_state[active_pipeline_key]["value"]
            pipeline_button_state[active_pipeline_key]["value"] = f"{base_label}（运行中 {elapsed_text}）"

    return pipeline_button_state, button_state
