"""训练页通用的文本与展示格式化函数。

这里尽量只保留纯函数：输入已经算好的值，输出用户可见的文本或 HTML，
这样复用和后续调整文案都会更轻一些。
"""

def format_duration(seconds: int) -> str:
    """把秒数格式化成适合中文界面展示的运行时长。"""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes > 0:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def format_duration_clock(seconds: int) -> str:
    """把秒数格式化成固定宽度的时钟样式字符串。"""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_stage_line(step_name: str, status: str, detail: str) -> str:
    """生成较完整的阶段说明文本。"""
    return f"{step_name}：{status}\n  {detail}"


def compact_stage_line(step_name: str, status: str, detail: str) -> str:
    """生成训练页里更紧凑的阶段摘要文本。"""
    if status in {"已完成", "已开始或已完成"}:
        icon = "✓"
    elif status in {"可执行"}:
        icon = "•"
    else:
        icon = "○"
    return f"{icon} {step_name}：{status}\n  {detail}"


def status_tone_tag(status: str):
    """把阶段状态映射成界面使用的颜色语气和圆点图标。"""
    if status in {"已完成", "已开始或已完成", "已就绪", "可用"}:
        return "ok", "●"
    if status in {"可执行", "等待上一步", "等待前置"}:
        return "warn", "●"
    return "missing", "●"


def render_dataset_import_result(message: str = "等待选择训练语音数据文件夹，选中后会自动导入到 dataset_raw。") -> str:
    """根据消息内容渲染“语音数据状态”卡片。"""
    text = (message or "").strip()
    if not text:
        text = "等待选择训练语音数据文件夹，选中后会自动导入到 dataset_raw。"
    tone = "#1f8f4c"
    if any(keyword in text for keyword in ["失败", "不存在", "禁止", "无法", "缺少"]):
        tone = "#c0392b"
    elif any(keyword in text for keyword in ["等待", "取消", "无需"]):
        tone = "#d97706"
    return (
        '<div class="dataset-import-result">'
        f'<div class="dataset-import-result__title"><span class="stage-dot" style="color:{tone};">●</span>语音数据状态</div>'
        f'<div class="dataset-import-result__body" style="color:{tone}; white-space: pre-line;">{text}</div>'
        '</div>'
    )


def render_pretrain_progress(ready: int, total: int, next_missing_label: str | None) -> str:
    """渲染训练前依赖与底模的进度摘要。"""
    missing = total - ready
    next_line = (
        '<div class="dependency-progress-hint">'
        '<span class="status-ok">训练前依赖与底模已就绪。</span>'
        '</div>'
    )
    if next_missing_label is not None:
        next_line = (
            '<div class="dependency-progress-hint">'
            '下一步补齐 '
            f'<span class="status-missing">{next_missing_label}</span>'
            '</div>'
        )
    return (
        '<div class="dependency-progress">'
        '<div class="dependency-progress-stats">'
        '<div class="dependency-progress-pill dependency-progress-pill--ok">'
        f'<span class="dependency-progress-pill__label">已完成</span><span class="dependency-progress-pill__value">{ready}</span>'
        '</div>'
        '<div class="dependency-progress-pill dependency-progress-pill--missing">'
        f'<span class="dependency-progress-pill__label">待补齐</span><span class="dependency-progress-pill__value">{missing}</span>'
        '</div>'
        f'<div class="dependency-progress-total">共 {total} 项</div>'
        '</div>'
        f'{next_line}'
        '</div>'
    )
