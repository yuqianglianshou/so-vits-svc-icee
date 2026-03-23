from __future__ import annotations
"""推理页本地模型扫描与记忆辅助函数。

这些函数只负责“本地模型目录”的扫描、刷新和上次选择记忆，
让入口文件只关心事件绑定，不再承担目录管理细节。
"""

import glob
import json
import os
import time
from pathlib import Path

import gradio as gr

LOCAL_MODEL_ROOT = './trained'


def last_selected_infer_model_path(local_model_root: str = LOCAL_MODEL_ROOT) -> Path:
    """返回记录上次选中本地模型的文件路径。"""
    return Path(local_model_root) / 'last_selected_infer_model.json'



def save_last_selected_infer_model(model_dir: str, local_model_root: str = LOCAL_MODEL_ROOT):
    """保存上次选中的本地模型目录。"""
    safe_value = (model_dir or '').strip()
    payload = {
        'model_dir': safe_value,
        'updated_at': int(time.time()),
    }
    target = last_selected_infer_model_path(local_model_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return safe_value



def load_last_selected_infer_model(local_model_root: str = LOCAL_MODEL_ROOT):
    """读取上次选中的本地模型目录。"""
    path = last_selected_infer_model_path(local_model_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    model_dir = str(payload.get('model_dir') or '').strip()
    return model_dir or None



def scan_local_models(local_model_root: str = LOCAL_MODEL_ROOT):
    """扫描 trained 目录下可直接加载的本地模型。"""
    res = []
    candidates = glob.glob(os.path.join(local_model_root, '**', '*.json'), recursive=True)
    candidates = set(os.path.dirname(candidate) for candidate in candidates)
    for candidate in candidates:
        jsons = glob.glob(os.path.join(candidate, '*.json'))
        pths = glob.glob(os.path.join(candidate, '*.pth'))
        if len(jsons) == 1 and len(pths) == 1:
            res.append(candidate)
    return res



def local_model_refresh_fn(local_model_root: str = LOCAL_MODEL_ROOT):
    """刷新本地模型下拉选项并尽量保留上次选择。"""
    choices = scan_local_models(local_model_root)
    remembered = load_last_selected_infer_model(local_model_root)
    value = remembered if remembered in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)



def persist_local_model_selection(local_model_selection, local_model_root: str = LOCAL_MODEL_ROOT):
    """在用户切换本地模型时持久化当前选择。"""
    if local_model_selection:
        save_last_selected_infer_model(local_model_selection, local_model_root)
    return gr.update()
