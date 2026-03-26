from __future__ import annotations

import gradio as gr

from train_ui.paths import default_train_dir_for_dataset, sanitize_dataset_name, sanitize_model_name


def refresh_live_task_panel(
    model_name: str,
    raw_dir: str,
    train_dir: str,
    *,
    active_task: dict,
    task_lifecycle_cache: dict,
    render_button_updates_fn,
    render_workspace_control_updates_fn,
    render_task_panel_snapshot_fn,
):
    proc = active_task["proc"]
    thread = active_task["thread"]
    is_running = (proc is not None and proc.poll() is None) or (thread is not None and thread.is_alive())
    exit_code = proc.returncode if proc is not None and proc.poll() is not None else None
    lifecycle_signature = (
        active_task["name"],
        active_task["pipeline_name"],
        active_task["stage_label"],
        is_running,
        exit_code,
    )
    if task_lifecycle_cache["signature"] != lifecycle_signature:
        task_lifecycle_cache["signature"] = lifecycle_signature
        task_lifecycle_cache["token"] = str(int(task_lifecycle_cache["token"]) + 1)
        task_refresh_token_update = gr.update(value=task_lifecycle_cache["token"])
    else:
        task_refresh_token_update = gr.skip()

    button_updates = render_button_updates_fn(model_name, raw_dir, train_dir)
    workspace_control_updates = render_workspace_control_updates_fn()
    return (
        *render_task_panel_snapshot_fn(model_name, raw_dir, train_dir),
        *button_updates,
        *workspace_control_updates,
        task_refresh_token_update,
    )


def refresh_dashboard(
    model_name: str,
    raw_dir: str,
    train_dir: str,
    *,
    render_model_workspace_summary_fn,
    dataset_file_list_label_fn,
    render_dataset_file_list_fn,
    render_stage_judgement_fn,
    render_preflight_check_fn,
    render_task_panel_snapshot_fn,
    load_model_batch_size_fn,
    render_button_updates_fn,
    render_workspace_control_updates_fn,
):
    button_updates = render_button_updates_fn(model_name, raw_dir, train_dir)
    workspace_control_updates = render_workspace_control_updates_fn()
    (
        stage_alert_text,
        runtime_banner_text,
        task_feedback,
        runtime_text,
        log_tail,
    ) = render_task_panel_snapshot_fn(model_name, raw_dir, train_dir)
    return (
        render_model_workspace_summary_fn(model_name),
        gr.update(label=dataset_file_list_label_fn(raw_dir)),
        render_dataset_file_list_fn(raw_dir),
        render_stage_judgement_fn(model_name, raw_dir, train_dir),
        render_preflight_check_fn(model_name, raw_dir, train_dir),
        stage_alert_text,
        runtime_banner_text,
        load_model_batch_size_fn(model_name),
        task_feedback,
        runtime_text,
        log_tail,
        *button_updates,
        *workspace_control_updates,
    )


def refresh_text_dashboard(
    model_name: str,
    raw_dir: str,
    train_dir: str,
    *,
    active_task: dict,
    render_model_workspace_summary_fn,
    dataset_file_list_label_fn,
    render_dataset_file_list_fn,
    render_stage_judgement_fn,
    render_preflight_check_fn,
    render_stage_alert_fn,
    current_task_feedback_fn,
    task_runtime_text_fn,
    tail_log_fn,
):
    log_path = active_task["log_path"]
    return (
        render_model_workspace_summary_fn(model_name),
        gr.update(label=dataset_file_list_label_fn(raw_dir)),
        render_dataset_file_list_fn(raw_dir),
        render_stage_judgement_fn(model_name, raw_dir, train_dir),
        render_preflight_check_fn(model_name, raw_dir, train_dir),
        render_stage_alert_fn(model_name, raw_dir, train_dir),
        current_task_feedback_fn(model_name, raw_dir, train_dir),
        task_runtime_text_fn(),
        tail_log_fn(log_path),
    )


def auto_refresh_dashboard(
    model_name: str,
    raw_dir: str,
    train_dir: str,
    *,
    active_task: dict,
    refresh_text_dashboard_fn,
    render_model_workspace_summary_fn,
    dataset_file_list_label_fn,
    render_dataset_file_list_fn,
    render_preflight_check_fn,
    render_stage_alert_fn,
    tail_log_fn,
):
    try:
        return refresh_text_dashboard_fn(model_name, raw_dir, train_dir)
    except Exception as exc:
        log_path = active_task["log_path"]
        safe_model_name = sanitize_model_name(model_name)
        safe_raw_dir = sanitize_dataset_name(raw_dir) or safe_model_name
        safe_train_dir = train_dir or default_train_dir_for_dataset(safe_raw_dir)
        fallback_message = f"自动刷新异常：{type(exc).__name__}: {exc}"
        return (
            render_model_workspace_summary_fn(safe_model_name),
            gr.update(label=dataset_file_list_label_fn(safe_raw_dir)),
            render_dataset_file_list_fn(safe_raw_dir),
            '<div class="stage-check-row"><div class="stage-check-title"><span class="stage-dot" style="color:#c0392b;">●</span>自动刷新失败，请稍后重试或手动点击刷新任务状态。</div></div>',
            render_preflight_check_fn(safe_model_name, safe_raw_dir, safe_train_dir),
            render_stage_alert_fn(safe_model_name, safe_raw_dir, safe_train_dir),
            fallback_message,
            fallback_message,
            tail_log_fn(log_path),
        )
