"""训练页任务状态面板的渲染辅助函数。

这些函数会被定时刷新频繁调用，所以这里尽量只负责把已经收集好的运行态
数据拼成最终显示文本，而不再做额外副作用操作。
"""

def build_task_feedback(
    stage_label: str,
    task_display: str,
    log_path: str,
    is_running: bool,
    exit_code: int | None,
    next_step_line: str,
) -> str:
    """生成任务反馈摘要文本。"""
    if is_running:
        return f"当前阶段：{stage_label}；任务：{task_display}；状态：运行中；日志：{log_path}"
    if exit_code == 0:
        return f"当前阶段：{stage_label}；任务：{task_display}；状态：已完成；日志：{log_path}；下一步：{next_step_line}"
    return (
        f"当前阶段：{stage_label}；任务：{task_display}；状态：已结束（退出码 {exit_code}）；"
        f"日志：{log_path}；建议：先看日志尾部再决定是否重试。"
    )


def build_stage_alert_text(task_name: str | None, stage_label: str, is_running: bool, succeeded: bool) -> str:
    """生成训练步骤上方那条紧凑的阶段状态文本。"""
    if task_name is None:
        return "当前没有运行中的任务"
    if is_running:
        return f"{stage_label} 进行中"
    if succeeded:
        return f"{stage_label} 已完成"
    return f"{stage_label} 执行异常"


def build_preflight_check_html(info_rows: list[str], blocker_lines: list[str]) -> str:
    """根据已整理好的信息项和阻塞项渲染开始前检查面板。"""
    if blocker_lines:
        blocker_rows = []
        for line in blocker_lines[:6]:
            blocker_rows.append(
                '<div class="stage-check-row">'
                f'<div class="stage-check-title"><span class="stage-dot" style="color:#c0392b;">●</span>{line}</div>'
                '</div>'
            )
        return (
            '<div class="preflight-group">'
            '<div class="preflight-heading">当前情况</div>'
            f'{"".join(info_rows)}'
            '<div class="preflight-heading">现在还缺</div>'
            f'{"".join(blocker_rows)}'
            '</div>'
        )

    return (
        '<div class="preflight-group">'
        '<div class="preflight-heading">当前情况</div>'
        f'{"".join(info_rows)}'
        '<div class="stage-next-step"><span style="color:#1f8f4c;">现在可以开始下一步。</span></div>'
        '</div>'
    )


def build_runtime_banner_text(
    started: float | None,
    task_display: str,
    is_running: bool,
    elapsed_clock_text: str,
    exit_code: int | None,
) -> str:
    """生成步骤按钮附近那张“运行提示”卡片的文本。"""
    if started is None:
        return "当前没有运行中的任务\n点击下方步骤后，这里会显示当前执行状态。"
    if is_running:
        return f"正在执行：{task_display}\n已进行 {elapsed_clock_text}"
    if exit_code == 0:
        return f"{task_display} 执行完毕\n可以继续下一步，相关按钮已恢复可点击。"
    if exit_code is not None:
        return f"{task_display} 执行异常\n请先查看下方日志摘要，再决定是否重试。"
    return f"{task_display} 已结束\n如果下方按钮状态还没同步，稍等 1 秒会自动刷新。"
