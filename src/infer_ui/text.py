from __future__ import annotations
"""推理页的状态文案与结果渲染辅助函数。

这里集中放“把当前状态整理成页面可展示文本/HTML”的纯函数，
避免把这些展示细节散落在 app_infer.py 入口文件里。
"""



def render_load_result_html(summary_text):
    """把模型加载结果整理成卡片 HTML。"""
    text = (summary_text or "").strip() or "先加载主模型和配置。"

    rows = []
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        tone = "ok"
        if "未加载" in line or "没有" in line:
            tone = "warn"
        if "失败" in line or "错误" in line:
            tone = "missing"
        color = "#1f8f4c" if tone == "ok" else "#d97706" if tone == "warn" else "#c0392b"
        rows.append(
            '<div class="readiness-row">'
            f'<div class="readiness-title"><span class="stage-dot" style="color:{color};">●</span>{line}</div>'
            '</div>'
        )
    return '<div class="stage-progress-box">' + "".join(rows) + '</div>'



def render_convert_result_html(message_text):
    """把转换结果整理成卡片 HTML。"""
    text = (message_text or "").strip() or "等待开始转换。"

    rows = []
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        tone = "ok"
        if line.startswith("提醒："):
            tone = "warn"
        elif "失败" in line or "错误" in line or "You need" in line:
            tone = "missing"
        color = "#1f8f4c" if tone == "ok" else "#d97706" if tone == "warn" else "#c0392b"
        rows.append(
            '<div class="readiness-row">'
            f'<div class="readiness-title"><span class="stage-dot" style="color:{color};">●</span>{line}</div>'
            '</div>'
        )
    return '<div class="stage-progress-box">' + "".join(rows) + '</div>'



def build_runtime_summary(device_name, speaker_name, quality_mode, vc_transform, cluster_ratio, k_step, has_diffusion, has_cluster):
    """导出当前推理运行参数摘要。"""
    return (
        f"设备：{device_name}\n"
        f"质量模式：{quality_mode}\n"
        f"目标音色：{speaker_name}\n"
        f"变调：{vc_transform}\n"
        f"特征检索混合比例：{cluster_ratio}\n"
        f"浅扩散步数：{k_step}\n"
        f"音质增强：{'已加载' if has_diffusion else '未加载'}\n"
        f"音色增强：{'已加载' if has_cluster else '未加载'}\n"
        f"输出格式：flac\n"
        f"F0 预测：rmvpe"
    )
