"""训练页面入口文件。

这个文件主要负责三件事：
1. 启动和管理训练 / 预处理类长任务，
2. 绑定训练页的 Gradio 组件与事件，
3. 串联已经拆出去的各个 UI 辅助模块。

已经拆出去的模块尽量只保留纯展示和格式化逻辑，这样主文件仍然能聚焦在
任务流程和页面事件本身。
"""

import json
import os
import platform
import ssl
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from huggingface_hub import hf_hub_download
from gradio_api_info_fallback import apply_gradio_4_api_info_patch
from path_utils import (
    BASE_MODEL_44K_DIFFUSION_DIR,
    BASE_MODEL_44K_DIR,
    ENCODER_DIR,
    VOCODER_DIR,
    get_contentvec_hf_path,
    get_diffusion_model_0_path,
    get_nsf_hifigan_config_path,
    get_nsf_hifigan_model_path,
    get_rmvpe_path,
    get_sovits_d0_path,
    get_sovits_g0_path,
)
from train_ui.paths import (
    default_train_dir_for_dataset,
    model_config_path,
    model_root_dir,
    model_workspace_path,
    resolve_raw_dataset_dir,
    sanitize_dataset_name,
    sanitize_model_name,
)
from train_ui.state import build_button_state_maps, collect_stage_state, render_stage_judgement_html
from train_ui.tasks import (
    launch_config as launch_config_task,
    launch_pipeline_prep as launch_pipeline_prep_task,
    launch_pipeline_train_main as launch_pipeline_train_main_task,
    launch_preprocess as launch_preprocess_task,
    launch_resample as launch_resample_task,
    launch_train as launch_train_task,
    launch_train_diff as launch_train_diff_task,
    launch_train_index as launch_train_index_task,
    start_pipeline as start_pipeline_task,
    start_task as start_task_process,
    stop_task as stop_task_process,
)
from train_ui.pretrain import (
    first_missing_pretrain_asset,
    is_pretrain_asset_ready,
    normalize_asset_key,
    ordered_pretrain_asset_keys,
    pretrain_asset_choices,
    render_pretrain_asset_guide as render_pretrain_asset_guide_html,
    render_pretrain_status as render_pretrain_status_html,
    resolve_uploaded_path,
)
from train_ui.workspace import (
    count_raw_dataset_wavs,
    count_training_wavs,
    dataset_file_list_label as build_dataset_file_list_label,
    has_raw_dataset_wavs,
    raw_dataset_display_name as build_raw_dataset_display_name,
    render_dataset_file_list as render_dataset_file_list_html,
    render_dataset_import_status_for_dataset as render_dataset_import_status_for_dataset_html,
    render_model_workspace_summary as render_model_workspace_summary_html,
)
from train_ui.panels import (
    build_preflight_check_html,
    build_runtime_banner_text,
    build_stage_alert_text,
    build_task_feedback,
)
from train_ui.text import (
    format_duration,
    format_duration_clock,
    render_dataset_import_result,
    render_pretrain_progress as render_pretrain_progress_html,
)
from utils import get_supported_speech_encoders

apply_gradio_4_api_info_patch()


ROOT = Path(__file__).resolve().parent
TRAIN_PAGE_CSS = (ROOT / "train_ui" / "page.css").read_text(encoding="utf-8")
TASK_LOG_DIR = ROOT / "logs" / "webui_tasks"
TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_TASK = {
    "name": None,
    "proc": None,
    "log_path": None,
    "cmd": None,
    "started_at": None,
    "thread": None,
    "pipeline_name": None,
    "stage_label": None,
    "stop_requested": False,
}

INFER_UI_STATE = {
    "proc": None,
    "port": None,
}

UI_NOTICE = {
    "message": None,
    "expires_at": 0.0,
}

TASK_LIFECYCLE_CACHE = {
    "signature": None,
    "token": "0",
}

DEFAULT_PREPROCESS_WORKERS = 6
DEFAULT_SPEECH_ENCODER = "vec768l12"

TASK_STAGE_LABELS = {
    "resample": "第 1 步：重采样",
    "preprocess_flist_config": "第 2 步：生成配置与文件列表",
    "preprocess_hubert_f0": "第 3 步：提取特征",
    "train_main": "第 4 步：主模型训练",
    "train_diff": "第 5 步：扩散训练",
    "train_index": "第 6 步：训练音色增强索引",
}

PIPELINE_LABELS = {
    "pipeline_prep": "一键执行 1-3 步",
    "pipeline_train_main": "一键执行到主模型训练",
}

RAW_DATASET_PARENT = ROOT / "dataset_raw"

PRETRAIN_ASSETS = {
    "contentvec_hf": {
        "label": "ContentVec HF 模型目录 contentvec_hf/",
        "target": ENCODER_DIR / "contentvec_hf",
        "accepted_names": {"contentvec_hf.zip"},
        "file_types": [".zip"],
        "required_files": ["config.json", "model.safetensors"],
        "download_links": [
            ("当前锁定来源", "https://huggingface.co/lengyue233/content-vec-best"),
        ],
        "purpose": "ContentVec 用来提取语音内容特征。当前项目已固定使用 Transformers/HF 路线，本地目录需包含 config.json 和 model.safetensors。",
        "is_archive": True,
    },
    "rmvpe": {
        "label": "RMVPE 模型 rmvpe.pt",
        "target": ENCODER_DIR / "rmvpe.pt",
        "accepted_names": {"rmvpe.pt", "model.pt", "rmvpe.zip"},
        "file_types": [".pt", ".zip"],
        "download_links": [
            ("RMVPE 发布页（zip）", "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"),
            ("RMVPE 直链（历史）", "https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt"),
        ],
        "purpose": "RMVPE 用来提取音高曲线，让模型知道每一句唱到多高。",
    },
    "sovits_g0": {
        "label": "So-VITS 预训练底模 G_0.pth",
        "target": BASE_MODEL_44K_DIR / "G_0.pth",
        "accepted_names": {"G_0.pth", "so-vits-pretrain.zip", "pretrain_model.zip"},
        "file_types": [".pth", ".zip"],
        "download_links": [
            ("G_0.pth（vec768l12）", "https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/vec768l12/G_0.pth"),
        ],
        "purpose": "G_0.pth 是主模型生成器底模，用来让主模型训练更稳定地起步。",
    },
    "sovits_d0": {
        "label": "So-VITS 预训练底模 D_0.pth",
        "target": BASE_MODEL_44K_DIR / "D_0.pth",
        "accepted_names": {"D_0.pth", "so-vits-pretrain.zip", "pretrain_model.zip"},
        "file_types": [".pth", ".zip"],
        "download_links": [
            ("D_0.pth（vec768l12）", "https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/vec768l12/D_0.pth"),
        ],
        "purpose": "D_0.pth 是主模型判别器底模，用来配合主模型训练。",
    },
    "diffusion_model_0": {
        "label": "扩散预训练底模 model_0.pt",
        "target": BASE_MODEL_44K_DIFFUSION_DIR / "model_0.pt",
        "accepted_names": {"model_0.pt", "so-vits-pretrain.zip", "pretrain_model.zip"},
        "file_types": [".pt", ".zip"],
        "download_links": [
            ("model_0.pt（diffusion/768l12）", "https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/diffusion/768l12/model_0.pt"),
        ],
        "purpose": "model_0.pt 是扩散模型底模，用来让音质增强训练更快收敛。",
    },
    "nsf_hifigan": {
        "label": "NSF-HIFIGAN 声码器包 nsf_hifigan_20221211.zip",
        "target": VOCODER_DIR / "nsf_hifigan",
        "accepted_names": {"nsf_hifigan_20221211.zip"},
        "file_types": [".zip"],
        "required_files": ["model", "config.json"],
        "download_links": [
            ("NSF-HIFIGAN zip", "https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip"),
        ],
        "purpose": "NSF-HIFIGAN 是声码器，扩散和增强链路会用它把特征还原成更自然的波形。",
        "is_archive": True,
    },
}

def is_rmvpe_asset_valid() -> bool:
    rmvpe_path = get_rmvpe_path()
    return rmvpe_path.exists() and rmvpe_path.is_file() and rmvpe_path.stat().st_size > 1024 * 1024

def render_pretrain_status():
    return render_pretrain_status_html(ROOT, PRETRAIN_ASSETS, get_rmvpe_path(), is_rmvpe_asset_valid)


def render_pretrain_asset_guide(asset_key):
    return render_pretrain_asset_guide_html(ROOT, asset_key, PRETRAIN_ASSETS, get_rmvpe_path(), is_rmvpe_asset_valid)


def render_pretrain_progress():
    total = len(PRETRAIN_ASSETS)
    ready = sum(1 for asset in PRETRAIN_ASSETS.values() if is_pretrain_asset_ready(asset, get_rmvpe_path(), is_rmvpe_asset_valid))
    next_missing = first_missing_pretrain_asset(PRETRAIN_ASSETS, get_rmvpe_path(), is_rmvpe_asset_valid)
    next_missing_label = PRETRAIN_ASSETS[next_missing]["label"] if next_missing is not None else None
    return render_pretrain_progress_html(ready, total, next_missing_label)


def pretrain_selector_update(selected_key: Optional[str] = None):
    ordered_keys = ordered_pretrain_asset_keys(PRETRAIN_ASSETS, get_rmvpe_path(), is_rmvpe_asset_valid)
    fallback = ordered_keys[0] if ordered_keys else "contentvec_hf"
    if selected_key not in PRETRAIN_ASSETS:
        selected_key = fallback
    return gr.update(choices=pretrain_asset_choices(PRETRAIN_ASSETS, get_rmvpe_path(), is_rmvpe_asset_valid), value=selected_key)


def pretrain_file_update(asset_key):
    asset_key = normalize_asset_key(asset_key, PRETRAIN_ASSETS)
    return gr.update(value=None, file_types=PRETRAIN_ASSETS[asset_key].get("file_types"))


def import_pretrain_asset(asset_key, uploaded_file):
    asset_key = normalize_asset_key(asset_key, PRETRAIN_ASSETS)
    asset = PRETRAIN_ASSETS[asset_key]
    source_path_str = resolve_uploaded_path(uploaded_file)
    if not source_path_str:
        return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)

    source_path = Path(source_path_str)
    if not source_path.exists():
        return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)

    allowed_types = {suffix.lower() for suffix in asset.get("file_types", [])}
    if allowed_types and source_path.suffix.lower() not in allowed_types:
        allowed = "、".join(sorted(allowed_types))
        return (
            render_pretrain_asset_guide(asset_key),
            render_pretrain_status(),
            gr.update(value=None),
        )

    if source_path.name not in asset["accepted_names"]:
        accepted = "、".join(sorted(asset["accepted_names"]))
        return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)

    zip_targets = {
        "rmvpe": {"rmvpe.pt", "model.pt"},
        "sovits_g0": {"G_0.pth"},
        "sovits_d0": {"D_0.pth"},
        "diffusion_model_0": {"model_0.pt"},
    }

    if asset_key in zip_targets and source_path.suffix.lower() == ".zip":
        asset["target"].parent.mkdir(parents=True, exist_ok=True)
        extracted = False
        with zipfile.ZipFile(source_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                member_name = Path(member.filename).name
                if member_name in zip_targets[asset_key]:
                    with zf.open(member) as src, asset["target"].open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted = True
                    break
        if not extracted:
            expected = " 或 ".join(sorted(zip_targets[asset_key]))
            return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)
        return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)

    if asset.get("is_archive"):
        asset["target"].mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_path, "r") as zf:
            zf.extractall(asset["target"])
        nested_dir = asset["target"] / "nsf_hifigan"
        if nested_dir.exists() and nested_dir.is_dir():
            for child in nested_dir.iterdir():
                shutil.move(str(child), asset["target"] / child.name)
            shutil.rmtree(nested_dir)
        return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)

    asset["target"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, asset["target"])
    return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)


def build_pretrain_guide_with_notice(asset_key, message: str, tone: str = "info"):
    """在依赖说明卡上方附加一条操作反馈，避免再额外占一个结果框。"""
    tone_map = {
        "info": ("#215f8b", "#eef6ff", "#c7def2"),
        "success": ("#1f8f4c", "#edf8f1", "#cce8d7"),
        "error": ("#b64737", "#fff1ee", "#f2cfc9"),
    }
    color, bg, border = tone_map.get(tone, tone_map["info"])
    notice = (
        f'<div style="margin-bottom:10px; padding:10px 12px; border:1px solid {border}; '
        f'background:{bg}; color:{color}; border-radius:12px; font-size:14px;">{message}</div>'
    )
    return notice + render_pretrain_asset_guide(asset_key)


def download_pretrain_asset(asset_key):
    """自动下载当前选中的训练前依赖。"""
    asset_key = normalize_asset_key(asset_key, PRETRAIN_ASSETS)
    asset = PRETRAIN_ASSETS[asset_key]

    if asset_key not in {"contentvec_hf", "rmvpe", "nsf_hifigan", "sovits_g0", "sovits_d0", "diffusion_model_0"}:
        message = "当前自动下载已支持 ContentVec HF、RMVPE、NSF-HIFIGAN 和底模。"
        return build_pretrain_guide_with_notice(asset_key, message, "info"), render_pretrain_status()

    target = asset["target"]
    download_url_map = {
        "contentvec_hf": {
            "config.json": {
                "repo_id": "lengyue233/content-vec-best",
                "revision": "ab04aa7067b99ee05cc82499bc64916b980a1967",
            },
            "model.safetensors": {
                "repo_id": "lengyue233/content-vec-best",
                "revision": "60a4eafc5775c9ff1f813fa544c0c8d3099898f2",
            },
        },
        "rmvpe": "https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt",
        "nsf_hifigan": "https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip",
        "sovits_g0": "https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/vec768l12/G_0.pth",
        "sovits_d0": "https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/vec768l12/D_0.pth",
        "diffusion_model_0": "https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/diffusion/768l12/model_0.pt",
    }

    tmp_path = None
    def _download_with_context(tmp_path: Path, context=None):
        with urllib.request.urlopen(download_url, context=context) as response, tmp_path.open("wb") as dst:
            shutil.copyfileobj(response, dst)

    try:
        if asset_key == "contentvec_hf":
            target.mkdir(parents=True, exist_ok=True)
            file_urls = download_url_map["contentvec_hf"]
            downloaded_files = []
            for filename, download_meta in file_urls.items():
                destination = target / filename
                try:
                    cached_path = hf_hub_download(
                        repo_id=download_meta["repo_id"],
                        filename=filename,
                        revision=download_meta["revision"],
                    )
                except Exception:
                    raise
                shutil.copyfile(cached_path, destination)
                downloaded_files.append(destination.name)
            message = (
                "ContentVec HF 已自动下载到 "
                f"{target.relative_to(ROOT).as_posix()}/（{', '.join(downloaded_files)}）。"
            )
            return build_pretrain_guide_with_notice(asset_key, message, "success"), render_pretrain_status()

        download_url = download_url_map[asset_key]
        temp_suffix_map = {
            "rmvpe": ".pt",
            "nsf_hifigan": ".zip",
            "sovits_g0": ".pth",
            "sovits_d0": ".pth",
            "diffusion_model_0": ".pt",
        }
        temp_suffix = temp_suffix_map[asset_key]
        temp_dir = str(target.parent if asset_key == "rmvpe" else target.parent)
        with tempfile.NamedTemporaryFile(prefix=f"{asset_key}_", suffix=temp_suffix, delete=False, dir=temp_dir) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _download_with_context(tmp_path)
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
                # macOS / 本地 Python 证书链不完整时，回退到不校验证书的下载方式，
                # 至少让训练页的自动下载能继续工作。
                insecure_context = ssl._create_unverified_context()
                _download_with_context(tmp_path, insecure_context)
            else:
                raise
        if asset_key in {"rmvpe", "sovits_g0", "sovits_d0", "diffusion_model_0"}:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_path), target)
            label = {
                "rmvpe": "RMVPE",
                "sovits_g0": "G_0.pth",
                "sovits_d0": "D_0.pth",
                "diffusion_model_0": "model_0.pt",
            }[asset_key]
            message = f"{label} 已自动下载到 {target.relative_to(ROOT).as_posix()}。"
        else:
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(target)
            nested_dir = target / "nsf_hifigan"
            if nested_dir.exists() and nested_dir.is_dir():
                for child in nested_dir.iterdir():
                    shutil.move(str(child), target / child.name)
                shutil.rmtree(nested_dir)
            tmp_path.unlink(missing_ok=True)
            message = f"NSF-HIFIGAN 已自动下载并解压到 {target.relative_to(ROOT).as_posix()}/。"
        return build_pretrain_guide_with_notice(asset_key, message, "success"), render_pretrain_status()
    except urllib.error.URLError as exc:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        asset_name = {
            "contentvec_hf": "ContentVec HF",
            "rmvpe": "RMVPE",
            "nsf_hifigan": "NSF-HIFIGAN",
            "sovits_g0": "G_0.pth",
            "sovits_d0": "D_0.pth",
            "diffusion_model_0": "model_0.pt",
        }[asset_key]
        message = f"{asset_name} 自动下载失败：{exc.reason}。你也可以继续使用手动导入。"
        return build_pretrain_guide_with_notice(asset_key, message, "error"), render_pretrain_status()
    except Exception as exc:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        asset_name = {
            "contentvec_hf": "ContentVec HF",
            "rmvpe": "RMVPE",
            "nsf_hifigan": "NSF-HIFIGAN",
            "sovits_g0": "G_0.pth",
            "sovits_d0": "D_0.pth",
            "diffusion_model_0": "model_0.pt",
        }[asset_key]
        message = f"{asset_name} 自动下载失败：{exc}"
        return build_pretrain_guide_with_notice(asset_key, message, "error"), render_pretrain_status()


def ensure_raw_dataset_parent():
    RAW_DATASET_PARENT.mkdir(parents=True, exist_ok=True)


def raw_dataset_display_name(dataset_name: str):
    return build_raw_dataset_display_name(sanitize_dataset_name(dataset_name) or "")


def ensure_model_workspace_dirs(model_name: str):
    root = model_root_dir(model_name)
    (root / "diffusion").mkdir(parents=True, exist_ok=True)
    (root / "filelists").mkdir(parents=True, exist_ok=True)
    return root


def save_model_workspace(model_name: str, dataset_name: str):
    model_name = sanitize_model_name(model_name)
    dataset_name = model_name
    train_dir = default_train_dir_for_dataset(dataset_name)
    ensure_model_workspace_dirs(model_name)
    payload = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "train_dir": train_dir,
        "updated_at": int(time.time()),
    }
    workspace_path = model_workspace_path(model_name)
    if workspace_path.exists():
        try:
            previous = json.loads(workspace_path.read_text(encoding="utf-8"))
            if "created_at" in previous:
                payload["created_at"] = previous["created_at"]
        except Exception:
            pass
    payload.setdefault("created_at", payload["updated_at"])
    workspace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_model_workspace(model_name: str):
    workspace_path = model_workspace_path(model_name)
    if not workspace_path.exists():
        return None
    try:
        return json.loads(workspace_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_workspace_dataset_name(model_name: str):
    return sanitize_model_name(model_name)


def scan_model_workspaces():
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for child in sorted(logs_dir.iterdir()):
        if not child.is_dir() or child.name == "webui_tasks":
            continue
        workspace = load_model_workspace(child.name)
        expected_dataset_name = infer_workspace_dataset_name(child.name)
        if workspace is None or workspace.get("dataset_name") != expected_dataset_name:
            workspace = save_model_workspace(child.name, expected_dataset_name)
        models.append(workspace["model_name"])
    if not models:
        models.append("default_model")
    return models


def last_selected_model_path() -> Path:
    return ROOT / "logs" / "last_selected_model.json"


def save_last_selected_model(model_name: str):
    safe_name = sanitize_model_name(model_name)
    payload = {
        "model_name": safe_name,
        "updated_at": int(time.time()),
    }
    last_selected_model_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return safe_name


def load_last_selected_model():
    path = last_selected_model_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model_name = sanitize_model_name(payload.get("model_name"))
    return model_name or None


def resolve_model_choice(model_name: str):
    safe_name = sanitize_model_name(model_name)
    candidates = scan_model_workspaces()
    if safe_name in candidates:
        return safe_name
    return candidates[0] if candidates else "default_model"


def has_active_task() -> bool:
    proc = ACTIVE_TASK["proc"]
    thread = ACTIVE_TASK["thread"]
    return (proc is not None and proc.poll() is None) or (thread is not None and thread.is_alive())


def active_task_block_message(action: str) -> str:
    task_display = ACTIVE_TASK["pipeline_name"] or ACTIVE_TASK["name"] or current_stage_label()
    return f"当前正在执行：{task_display}；请先停止当前任务，再{action}。"


def show_active_task_block_dialog(action: str):
    return (
        active_task_block_message(action),
        gr.update(visible=True),
    )


def render_workspace_control_updates():
    locked = has_active_task()
    return (
        gr.update(interactive=not locked),
        gr.update(interactive=not locked),
        gr.update(interactive=not locked),
        gr.update(interactive=not locked),
    )


def normalize_speech_encoder_choice(speech_encoder: str):
    supported = set(get_supported_speech_encoders())
    choice = (speech_encoder or "").strip()
    choice = {
        "vec768l12_hf": DEFAULT_SPEECH_ENCODER,
        "vec768l12_fairseq": DEFAULT_SPEECH_ENCODER,
    }.get(choice, choice)
    if choice in supported:
        return choice
    return DEFAULT_SPEECH_ENCODER


def speech_encoder_value_update(speech_encoder: str):
    normalized = normalize_speech_encoder_choice(speech_encoder)
    return gr.update(value=normalized)


def load_model_speech_encoder(model_name: str):
    config_path = model_config_path(sanitize_model_name(model_name))
    if not config_path.exists():
        return DEFAULT_SPEECH_ENCODER
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_SPEECH_ENCODER
    return normalize_speech_encoder_choice(config.get("model", {}).get("speech_encoder", DEFAULT_SPEECH_ENCODER))


def render_model_workspace_summary(model_name: str):
    model_name = sanitize_model_name(model_name)
    workspace = load_model_workspace(model_name)
    dataset_name = model_name
    train_dir = workspace.get("train_dir", default_train_dir_for_dataset(dataset_name))
    return render_model_workspace_summary_html(
        ROOT,
        model_name,
        workspace,
        dataset_name,
        train_dir,
        resolve_raw_dataset_dir(dataset_name),
    )


def create_model_workspace(new_model_name: str, current_dataset_name: str):
    """新建模型工作区，或按模型名重新选中已有工作区。

    训练页默认把“模型名”和“绑定的数据目录名”保持为一一对应，所以这里在
    创建工作区时也会同步初始化对应的数据目录绑定关系。
    """
    if has_active_task():
        current_model = sanitize_model_name(current_dataset_name) or "default_model"
        current_dataset = sanitize_dataset_name(current_dataset_name) or current_model
        return (
            gr.update(value=current_model),
            gr.update(value=current_model),
            gr.update(value=current_model),
            gr.update(value=current_dataset),
            gr.update(value=default_train_dir_for_dataset(current_dataset)),
            gr.update(value=suggest_model_name_for_dataset(current_dataset)),
            speech_encoder_value_update(load_model_speech_encoder(current_model)),
            render_model_workspace_summary(current_model),
            render_dataset_import_result(active_task_block_message("新建模型工作区")),
        )
    model_name = sanitize_model_name(new_model_name)
    dataset_name = model_name
    workspace_exists = model_workspace_path(model_name).exists()
    save_model_workspace(model_name, dataset_name)
    save_last_selected_model(model_name)
    return (
        gr.update(choices=scan_model_workspaces(), value=model_name),
        gr.update(value=model_name),
        gr.update(value=model_name),
        gr.update(value=dataset_name),
        gr.update(value=default_train_dir_for_dataset(dataset_name)),
        gr.update(value=suggest_model_name_for_dataset(dataset_name)),
        speech_encoder_value_update(load_model_speech_encoder(model_name)),
        render_model_workspace_summary(model_name),
        render_dataset_import_result(
            f"{'已切换到已有模型工作区' if workspace_exists else '已新建模型工作区'}：{model_name}；绑定模型数据目录：{dataset_name}"
        ),
    )


def switch_model_workspace(selected_model_name: str, current_model_name: str, current_dataset_name: str, current_train_dir: str, current_sync_token: str):
    """把页面上下文切换到另一个模型工作区。

    这个方法直接由“切换模型”下拉触发，所以必须在任务运行中尽早拦截，
    避免页面先短暂切过去、随后又被恢复，造成错觉。
    """
    if has_active_task():
        current_model = sanitize_model_name(ACTIVE_TASK.get("name") or "")
        return (
            gr.update(value=sanitize_model_name(current_model_name) or "default_model"),
            gr.update(value=sanitize_model_name(current_model_name) or "default_model"),
            gr.update(value=sanitize_model_name(current_model_name) or "default_model"),
            gr.update(value=sanitize_dataset_name(current_dataset_name) or "default_dataset"),
            gr.update(value=current_train_dir or default_train_dir_for_dataset(current_dataset_name)),
            gr.update(value=current_sync_token),
            speech_encoder_value_update(load_model_speech_encoder(current_model_name)),
            render_model_workspace_summary(sanitize_model_name(current_model_name) or "default_model"),
            render_dataset_import_result(active_task_block_message("切换模型")),
            *show_active_task_block_dialog("切换模型"),
        )
    model_name = resolve_model_choice(selected_model_name)
    dataset_name = infer_workspace_dataset_name(model_name)
    save_model_workspace(model_name, dataset_name)
    save_last_selected_model(model_name)
    raw_dir_exists = (ROOT / resolve_raw_dataset_dir(dataset_name)).exists()
    return (
        gr.update(choices=scan_model_workspaces(), value=model_name),
        gr.update(value=model_name),
        gr.update(value=model_name),
        gr.update(value=dataset_name),
        gr.update(value=default_train_dir_for_dataset(dataset_name)),
        gr.update(value=suggest_model_name_for_dataset(dataset_name)),
        speech_encoder_value_update(load_model_speech_encoder(model_name)),
        render_model_workspace_summary(model_name),
        render_dataset_import_result(
            f"已切换到模型工作区：{model_name}；绑定模型数据目录：{dataset_name}；数据目录：{'已存在' if raw_dir_exists else '未找到，请导入 wav 数据'}"
        ),
        gr.update(),
        gr.update(visible=False),
    )


def prepare_delete_model_workspace(selected_model_name: str):
    """生成“删除模型工作区”确认弹窗需要的提示内容。"""
    if has_active_task():
        return (
            "",
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            render_dataset_import_result(active_task_block_message("删除模型工作区")),
            *show_active_task_block_dialog("删除模型工作区"),
        )
    model_name = resolve_model_choice(selected_model_name)
    model_dir = model_root_dir(model_name)
    dataset_name = infer_workspace_dataset_name(model_name)
    raw_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    train_dir = ROOT / default_train_dir_for_dataset(dataset_name)
    if not model_dir.exists() and not raw_dir.exists() and not train_dir.exists():
        return (
            f"**logs/{model_name}**、**{resolve_raw_dataset_dir(dataset_name).as_posix()}**、**{default_train_dir_for_dataset(dataset_name)}** 都不存在，无需删除。",
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
        )
    message = (
        f"确认删除：{model_dir.relative_to(ROOT).as_posix()}\n\n"
        f"当前模型：{model_name}\n"
        f"绑定模型数据目录：{dataset_name}\n\n"
        "这个操作会一并删除以下内容：\n"
        f"- logs/{model_name}\n"
        f"- {resolve_raw_dataset_dir(dataset_name).as_posix()}\n"
        f"- {default_train_dir_for_dataset(dataset_name)}"
    )
    return (
        message,
        gr.update(visible=True),
        gr.update(value=model_name),
        gr.update(value="workspace"),
        gr.update(),
        gr.update(),
        gr.update(visible=False),
    )


def delete_model_workspace(selected_model_name: str):
    """删除模型工作区及其绑定目录，并切回一个可用的工作区。"""
    model_name = resolve_model_choice(selected_model_name)
    model_dir = model_root_dir(model_name)
    dataset_name = infer_workspace_dataset_name(model_name)
    raw_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    train_dir = ROOT / default_train_dir_for_dataset(dataset_name)
    deleted_targets = []

    for target in (model_dir, raw_dir, train_dir):
        if target.exists():
            shutil.rmtree(target)
            deleted_targets.append(target.relative_to(ROOT).as_posix())

    if not deleted_targets:
        next_model = resolve_model_choice("default_model")
        next_dataset = infer_workspace_dataset_name(next_model)
        save_last_selected_model(next_model)
        return (
            gr.update(choices=scan_model_workspaces(), value=next_model),
            gr.update(value=next_model),
            gr.update(value=next_model),
            gr.update(value=next_dataset),
            gr.update(value=default_train_dir_for_dataset(next_dataset)),
            gr.update(value=suggest_model_name_for_dataset(next_dataset)),
            speech_encoder_value_update(load_model_speech_encoder(next_model)),
            render_model_workspace_summary(next_model),
            render_dataset_import_result(
                f"logs/{model_name}、{resolve_raw_dataset_dir(dataset_name).as_posix()}、{default_train_dir_for_dataset(dataset_name)} 都不存在，无需删除。"
            ),
        )
    remaining_models = scan_model_workspaces()
    next_model = remaining_models[0] if remaining_models else "default_model"
    next_dataset = infer_workspace_dataset_name(next_model)
    save_model_workspace(next_model, next_dataset)
    save_last_selected_model(next_model)
    return (
        gr.update(choices=remaining_models, value=next_model),
        gr.update(value=next_model),
        gr.update(value=next_model),
        gr.update(value=next_dataset),
        gr.update(value=default_train_dir_for_dataset(next_dataset)),
        gr.update(value=suggest_model_name_for_dataset(next_dataset)),
        speech_encoder_value_update(load_model_speech_encoder(next_model)),
        render_model_workspace_summary(next_model),
        render_dataset_import_result(
            f"已删除模型相关目录：{'；'.join(deleted_targets)}；当前已切换到：{next_model}"
        ),
    )


def confirm_delete_action(action_kind: str, target_name: str):
    action_kind = (action_kind or "").strip()
    if action_kind == "workspace":
        (
            workspace_selector_update,
            workspace_new_name_update,
            train_model_name_update,
            dataset_source_dir_update,
            dataset_train_dir_update,
            model_name_sync_token_update,
            speech_encoder_selector_update,
            workspace_summary_html,
            dataset_import_html,
        ) = delete_model_workspace(target_name)
        return (
            dataset_import_html,
            dataset_source_dir_update,
            gr.update(),
            gr.update(),
            gr.update(),
            dataset_train_dir_update,
            train_model_name_update,
            workspace_selector_update,
            workspace_new_name_update,
            model_name_sync_token_update,
            speech_encoder_selector_update,
            workspace_summary_html,
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
        )

    (
        dataset_import_html,
        dataset_source_dir_update,
        dataset_file_list_html,
        dataset_new_dir_name_update,
        dataset_train_dir_update,
        train_model_name_update,
    ) = delete_dataset_directory(target_name)
    return (
        dataset_import_html,
        dataset_source_dir_update,
        dataset_file_list_html,
        dataset_new_dir_name_update,
        dataset_train_dir_update,
        train_model_name_update,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""),
    )


def resolve_dataset_choice(dataset_name: str):
    candidates = scan_dataset_candidates()
    safe_name = sanitize_dataset_name(dataset_name)
    if safe_name in candidates:
        return safe_name
    return candidates[0] if candidates else "default_dataset"

def suggest_model_name_for_dataset(dataset_name: str):
    return sanitize_model_name(dataset_name or "") or "default_model"


def track_model_name_mode(model_name: str, dataset_name: str):
    model_name = (model_name or "").strip()
    suggested_name = suggest_model_name_for_dataset(dataset_name)
    if not model_name or model_name == suggested_name:
        return suggested_name
    return "__manual__"


def bind_workspace_dataset(model_name: str, dataset_name: str):
    model_name = sanitize_model_name(model_name)
    save_model_workspace(model_name, model_name)
    return render_model_workspace_summary(model_name)


def scan_dataset_candidates():
    ensure_raw_dataset_parent()
    candidates = []
    for child in sorted(RAW_DATASET_PARENT.iterdir()):
        if child.is_dir() and has_raw_dataset_wavs(child):
            candidates.append(child.name)
    if not candidates:
        candidates.append("default_dataset")
    return candidates

def suggest_next_dataset_name():
    ensure_raw_dataset_parent()
    index = 1
    while True:
        candidate = f"speak {index}"
        if not (RAW_DATASET_PARENT / candidate).exists():
            return candidate
        index += 1


def infer_uploaded_dataset_name(uploaded_files):
    if not uploaded_files:
        return None
    file_items = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    parent_names = []
    ignored_names = {"", ".", "..", "tmp", "temp", "gradio"}
    for item in file_items:
        original_name = getattr(item, "orig_name", "") or ""
        parent_name = Path(original_name).parent.name if original_name else ""
        if parent_name.lower() in ignored_names:
            continue
        if len(parent_name) >= 24 and all(ch in "0123456789abcdef" for ch in parent_name.lower()):
            continue
        parent_names.append(parent_name)
    if not parent_names:
        return None
    unique_names = {name for name in parent_names if name}
    if len(unique_names) == 1:
        return unique_names.pop()
    return None


def import_dataset_directory(uploaded_files, dataset_name: str):
    dataset_name = sanitize_dataset_name(dataset_name) or "default_dataset"
    if not uploaded_files:
        return render_dataset_import_result("请先选择一个数据集文件夹。"), gr.update(value=dataset_name)

    target_relative_dir = resolve_raw_dataset_dir(dataset_name)
    target_root = ROOT / target_relative_dir
    target_root.mkdir(parents=True, exist_ok=True)

    file_items = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    copied = 0
    overwritten = 0
    for item in file_items:
        source_path = Path(resolve_uploaded_path(item))
        original_name = getattr(item, "orig_name", "") or ""
        file_name = Path(original_name).name if original_name else source_path.name
        if not file_name.lower().endswith(".wav"):
            continue
        destination = target_root / file_name
        if destination.exists():
            overwritten += 1
        shutil.copyfile(source_path, destination)
        copied += 1

    if copied == 0:
        return render_dataset_import_result("导入失败：没有检测到 wav 文件。请选择一个内部直接包含 `.wav` 文件的模型数据文件夹。"), gr.update(value=dataset_name)

    return (
        render_dataset_import_result(
            f"本次已导入到：{target_relative_dir.as_posix()}；当前模型数据目录：{raw_dataset_display_name(dataset_name)}；导入音频数：{copied}；覆盖同名文件数：{overwritten}"
        ),
        gr.update(value=dataset_name),
    )


def refresh_dataset_name_suggestion():
    return gr.update(value=suggest_next_dataset_name())


def render_dataset_file_list(dataset_name: str):
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    return render_dataset_file_list_html(ROOT, dataset_dir)


def dataset_file_list_label(dataset_name: str):
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    _, wav_count = count_raw_dataset_wavs(dataset_dir)
    return build_dataset_file_list_label(wav_count)


def render_dataset_import_status_for_dataset(dataset_name: str):
    dataset_name = sanitize_dataset_name(dataset_name) or "default_dataset"
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    return render_dataset_import_status_for_dataset_html(ROOT, dataset_dir)


def delete_dataset_directory(dataset_name: str):
    dataset_name = sanitize_dataset_name(dataset_name) or "default_dataset"
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    current_train_dir = default_train_dir_for_dataset(dataset_name)
    if not dataset_dir.exists():
        return (
            render_dataset_import_result(f"{dataset_dir.relative_to(ROOT).as_posix()} 不存在，无需删除。"),
            gr.update(value=dataset_name),
            render_dataset_file_list(dataset_name),
            gr.update(value=dataset_name),
            gr.update(value=current_train_dir),
            gr.update(value=dataset_name),
        )

    if dataset_dir == RAW_DATASET_PARENT:
        return (
            render_dataset_import_result("禁止删除 dataset_raw 父目录。"),
            gr.update(value=dataset_name),
            render_dataset_file_list(dataset_name),
            gr.update(value=dataset_name),
            gr.update(value=current_train_dir),
            gr.update(value=dataset_name),
        )

    shutil.rmtree(dataset_dir)
    return (
        render_dataset_import_result(f"已删除模型数据目录：{dataset_dir.relative_to(ROOT).as_posix()}"),
        gr.update(value=dataset_name),
        render_dataset_file_list(dataset_name),
        gr.update(value=dataset_name),
        gr.update(value=current_train_dir),
        gr.update(value=dataset_name),
    )


def prepare_delete_dataset(dataset_name: str):
    if has_active_task():
        return (
            "",
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            *show_active_task_block_dialog("删除当前模型数据目录"),
        )
    dataset_dir = ROOT / resolve_raw_dataset_dir(dataset_name)
    if not dataset_dir.exists():
        return (
            f"**{dataset_dir.relative_to(ROOT).as_posix()}** 不存在，无需删除。",
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(),
            gr.update(visible=False),
        )
    wav_count = len([path for path in dataset_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"])
    message = (
        f"确认删除：{dataset_dir.relative_to(ROOT).as_posix()}\n\n"
        f"该目录下共有 {wav_count} 个 wav 文件。\n"
        "删除后不可恢复。"
    )
    return (
        message,
        gr.update(visible=True),
        gr.update(value=dataset_name),
        gr.update(value="dataset"),
        gr.update(),
        gr.update(visible=False),
    )


def detect_file(path_str: str):
    path = ROOT / path_str
    return "已就绪" if path.exists() else "缺失"


def append_pipeline_log(log_path: Path, message: str):
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message.rstrip() + "\n")
def render_stage_judgement(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "dataset/44k"):
    state = collect_stage_state(model_name, raw_dir, train_dir)
    return render_stage_judgement_html(state)


def render_button_updates(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "dataset/44k"):
    stage_state = collect_stage_state(model_name, raw_dir, train_dir)
    task_name = ACTIVE_TASK["name"]
    pipeline_name = ACTIVE_TASK["pipeline_name"]
    proc = ACTIVE_TASK["proc"]
    thread = ACTIVE_TASK["thread"]
    started_at = ACTIVE_TASK["started_at"] or time.time()
    is_running = (proc is not None and proc.poll() is None) or (thread is not None and thread.is_alive())
    elapsed_seconds = max(0, int(time.time() - started_at))
    elapsed_text = format_duration(elapsed_seconds)
    pipeline_button_state, button_state = build_button_state_maps(
        stage_state,
        task_name,
        pipeline_name,
        is_running,
        elapsed_text,
    )

    return (
        gr.update(value=pipeline_button_state["pipeline_prep"]["value"], interactive=pipeline_button_state["pipeline_prep"]["interactive"]),
        gr.update(value=pipeline_button_state["pipeline_train"]["value"], interactive=pipeline_button_state["pipeline_train"]["interactive"]),
        gr.update(value=button_state["resample"]["value"], interactive=button_state["resample"]["interactive"]),
        gr.update(value=button_state["config"]["value"], interactive=button_state["config"]["interactive"]),
        gr.update(value=button_state["preprocess"]["value"], interactive=button_state["preprocess"]["interactive"]),
        gr.update(value=button_state["train"]["value"], interactive=button_state["train"]["interactive"]),
        gr.update(value=button_state["train_diff"]["value"], interactive=button_state["train_diff"]["interactive"]),
        gr.update(value=button_state["train_index"]["value"], interactive=button_state["train_index"]["interactive"]),
    )


def current_stage_label():
    if ACTIVE_TASK["stage_label"]:
        return ACTIVE_TASK["stage_label"]
    task_name = ACTIVE_TASK["name"]
    if not task_name:
        return "当前没有运行中的任务。"
    return TASK_STAGE_LABELS.get(task_name, task_name)


def current_task_feedback(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "dataset/44k"):
    proc = ACTIVE_TASK["proc"]
    task_name = ACTIVE_TASK["name"]
    log_path = ACTIVE_TASK["log_path"]
    pipeline_name = ACTIVE_TASK["pipeline_name"]
    thread = ACTIVE_TASK["thread"]
    if task_name is None and pipeline_name is None:
        return "等待操作。"

    task_display = PIPELINE_LABELS.get(pipeline_name, pipeline_name) if pipeline_name else task_name
    stage_label = current_stage_label()
    is_running = thread is not None and thread.is_alive() and (proc is None or proc.poll() is None)
    if is_running:
        return build_task_feedback(stage_label, task_display, str(log_path), True, None, "")

    exit_code = proc.returncode if proc is not None else None
    stage_state = collect_stage_state(model_name, raw_dir, train_dir)
    next_step_line = stage_state["next_step"]
    return build_task_feedback(stage_label, task_display, str(log_path), False, exit_code, next_step_line)


def render_stage_alert(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "dataset/44k"):
    proc = ACTIVE_TASK["proc"]
    task_name = ACTIVE_TASK["name"]
    stage_label = current_stage_label()
    is_running = proc is not None and proc.poll() is None
    succeeded = proc is not None and proc.returncode == 0
    return build_stage_alert_text(task_name, stage_label, is_running, succeeded)


def detect_cuda_status():
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


def render_preflight_check(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "dataset/44k"):
    model_name = sanitize_model_name(model_name)
    stage_state = collect_stage_state(model_name, raw_dir, train_dir)
    raw_relative_dir = resolve_raw_dataset_dir(raw_dir)
    _, raw_wavs = count_raw_dataset_wavs(ROOT / raw_relative_dir)
    train_speakers, train_wavs = count_training_wavs(ROOT / train_dir)
    cuda_status = detect_cuda_status()

    train_requirements = []
    if "CUDA 可用" not in cuda_status:
        train_requirements.append("主模型训练和扩散训练需要 Windows/Linux + NVIDIA GPU + CUDA 环境。")
    if not get_sovits_g0_path().exists():
        train_requirements.append(f"缺少 So-VITS 生成器底模：{get_sovits_g0_path().relative_to(ROOT).as_posix()}")
    if not get_sovits_d0_path().exists():
        train_requirements.append(f"缺少 So-VITS 判别器底模：{get_sovits_d0_path().relative_to(ROOT).as_posix()}")
    if not get_diffusion_model_0_path().exists():
        train_requirements.append(f"缺少扩散底模：{get_diffusion_model_0_path().relative_to(ROOT).as_posix()}")
    if not get_rmvpe_path().exists():
        train_requirements.append(f"缺少 RMVPE 预训练文件：{get_rmvpe_path().relative_to(ROOT).as_posix()}")
    elif not is_rmvpe_asset_valid():
        train_requirements.append(f"RMVPE 预训练文件已损坏，请重新导入：{get_rmvpe_path().relative_to(ROOT).as_posix()}")
    contentvec_hf_dir = get_contentvec_hf_path()
    if not (
        (contentvec_hf_dir / "config.json").exists()
        and (contentvec_hf_dir / "model.safetensors").exists()
    ):
        train_requirements.append(
            f"缺少 ContentVec HF 模型目录：{contentvec_hf_dir.relative_to(ROOT).as_posix()}/"
        )
    if not (get_nsf_hifigan_model_path().exists() and get_nsf_hifigan_config_path().exists()):
        train_requirements.append(f"缺少 NSF-HIFIGAN 声码器：{get_nsf_hifigan_model_path().parent.relative_to(ROOT).as_posix()}/")
    if raw_wavs == 0:
        train_requirements.append(f"{raw_relative_dir.as_posix()} 为空，无法开始完整训练流程。")
    if train_wavs > 0 and train_speakers != 1:
        train_requirements.append(f"处理后数据目录中只能保留 1 个训练数据子目录，当前检测到 {train_speakers} 个。")

    summary = stage_state["summary"]
    for needle, message in (
        ("3. 提取特征：等待上一步", "特征预处理还不能开始，先补齐配置与文件列表。"),
        ("4. 主模型训练：等待上一步", "主模型训练还不能开始，先完成特征预处理。"),
        ("5. 扩散训练：等待上一步", "扩散训练还不能开始，先得到可用的主模型结果。"),
        ("6. 训练音色增强索引：等待上一步", "音色增强索引还不能开始，先得到可用的主模型结果。"),
    ):
        if needle in summary:
            train_requirements.append(message)

    next_step_line = stage_state["next_step"]

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
        (
            f"处理后数据：{train_dir}，{train_wavs} 个 wav",
            "#1f8f4c" if train_wavs > 0 else "#d97706",
        ),
    ]

    info_rows = []
    for line, color in head:
        info_rows.append(
            '<div class="stage-check-row">'
            f'<div class="stage-check-title"><span class="stage-dot" style="color:{color};">●</span>{line}</div>'
            '</div>'
        )

    return build_preflight_check_html(info_rows, train_requirements)


def task_runtime_text():
    proc = ACTIVE_TASK["proc"]
    thread = ACTIVE_TASK["thread"]
    pipeline_name = ACTIVE_TASK["pipeline_name"]
    if proc is None and not (thread is not None and thread.is_alive()):
        return "当前没有运行中的任务。"
    started = ACTIVE_TASK["started_at"] or time.time()
    elapsed = int(time.time() - started)
    elapsed_text = format_duration(elapsed)
    if proc is None and thread is not None and thread.is_alive():
        status = "运行中（流程编排中）"
    else:
        status = "运行中" if proc.poll() is None else f"已结束（退出码 {proc.returncode}）"
    task_display = PIPELINE_LABELS.get(pipeline_name, pipeline_name) if pipeline_name else ACTIVE_TASK["name"]
    return (
        f"任务：{task_display}\n"
        f"阶段：{current_stage_label()}\n"
        f"状态：{status}\n"
        f"已运行：{elapsed_text}\n"
        f"日志：{ACTIVE_TASK['log_path']}\n"
        f"命令：{' '.join(ACTIVE_TASK['cmd']) if ACTIVE_TASK['cmd'] else '等待当前子步骤启动'}"
    )


def render_runtime_banner():
    proc = ACTIVE_TASK["proc"]
    thread = ACTIVE_TASK["thread"]
    pipeline_name = ACTIVE_TASK["pipeline_name"]
    started = ACTIVE_TASK["started_at"]
    is_running = (proc is not None and proc.poll() is None) or (thread is not None and thread.is_alive())

    task_display = PIPELINE_LABELS.get(pipeline_name, pipeline_name) if pipeline_name else current_stage_label()
    elapsed_text = format_duration_clock(max(0, int(time.time() - started))) if started is not None else ""
    exit_code = proc.returncode if proc is not None else None
    return build_runtime_banner_text(started, task_display, is_running, elapsed_text, exit_code)


def tail_log(path: Path, max_lines: int = 80):
    if path is None or not path.exists():
        return "日志文件尚不存在。"
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:]) if lines else "日志暂时为空。"


def extract_log_highlights(path: Path, max_lines: int = 30):
    if path is None or not path.exists():
        return "当前还没有重点提示。"
    keywords = (
        "error",
        "exception",
        "traceback",
        "warning",
        "failed",
        "runtimeerror",
        "assert",
        "nan",
        "inf",
        "oom",
        "cuda",
    )
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    matched = [line for line in lines if any(keyword in line.lower() for keyword in keywords)]
    if not matched:
        return "最近日志里没有明显的错误或警告。"
    return "".join(matched[-max_lines:])


def render_auto_batch_probe_summary(path: Path):
    """显示自动 batch size 的极限值和实际推荐值。"""
    if path is None or not path.exists():
        return "当前还没有自动 batch size 结果。"

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    max_supported_matches = re.findall(r"极限每卡 batch size：(\d+)", content)
    recommended_matches = re.findall(r"推荐值：(\d+)", content)

    if max_supported_matches and recommended_matches:
        return (
            f"极限每卡 batch size：{max_supported_matches[-1]}\n"
            f"实际推荐 batch size：{recommended_matches[-1]}"
        )

    if "[AutoBatch]" in content:
        return "自动 batch size 探测进行中，等待结果..."

    return "当前还没有自动 batch size 结果。"


def render_task_panel_snapshot(model_name: str, raw_dir: str, train_dir: str):
    log_path = ACTIVE_TASK["log_path"]
    return (
        render_stage_alert(model_name, raw_dir, train_dir),
        render_runtime_banner(),
        render_auto_batch_probe_summary(log_path),
        current_task_feedback(model_name, raw_dir, train_dir),
        task_runtime_text(),
        render_log_highlights(log_path),
        tail_log(log_path),
    )


def refresh_live_task_panel(model_name: str, raw_dir: str, train_dir: str):
    """只刷新定时器负责的任务状态区域。"""
    proc = ACTIVE_TASK["proc"]
    thread = ACTIVE_TASK["thread"]
    is_running = (proc is not None and proc.poll() is None) or (thread is not None and thread.is_alive())
    exit_code = proc.returncode if proc is not None and proc.poll() is not None else None
    lifecycle_signature = (
        ACTIVE_TASK["name"],
        ACTIVE_TASK["pipeline_name"],
        ACTIVE_TASK["stage_label"],
        is_running,
        exit_code,
    )
    if TASK_LIFECYCLE_CACHE["signature"] != lifecycle_signature:
        TASK_LIFECYCLE_CACHE["signature"] = lifecycle_signature
        TASK_LIFECYCLE_CACHE["token"] = str(int(TASK_LIFECYCLE_CACHE["token"]) + 1)
        task_refresh_token_update = gr.update(value=TASK_LIFECYCLE_CACHE["token"])
    else:
        task_refresh_token_update = gr.skip()

    button_updates = render_button_updates(model_name, raw_dir, train_dir)
    workspace_control_updates = render_workspace_control_updates()
    return (
        *render_task_panel_snapshot(model_name, raw_dir, train_dir),
        *button_updates,
        *workspace_control_updates,
        task_refresh_token_update,
    )


def refresh_dashboard(model_name: str, raw_dir: str, train_dir: str):
    """在用户显式操作后刷新训练页的大部分状态快照。"""
    button_updates = render_button_updates(model_name, raw_dir, train_dir)
    workspace_control_updates = render_workspace_control_updates()
    return (
        render_model_workspace_summary(model_name),
        gr.update(label=dataset_file_list_label(raw_dir)),
        render_dataset_file_list(raw_dir),
        render_stage_judgement(model_name, raw_dir, train_dir),
        render_preflight_check(model_name, raw_dir, train_dir),
        *render_task_panel_snapshot(model_name, raw_dir, train_dir),
        *button_updates,
        *workspace_control_updates,
    )


def refresh_text_dashboard(model_name: str, raw_dir: str, train_dir: str):
    log_path = ACTIVE_TASK["log_path"]
    return (
        render_model_workspace_summary(model_name),
        gr.update(label=dataset_file_list_label(raw_dir)),
        render_dataset_file_list(raw_dir),
        render_stage_judgement(model_name, raw_dir, train_dir),
        render_preflight_check(model_name, raw_dir, train_dir),
        render_stage_alert(model_name, raw_dir, train_dir),
        render_auto_batch_probe_summary(log_path),
        current_task_feedback(model_name, raw_dir, train_dir),
        task_runtime_text(),
        render_log_highlights(log_path),
        tail_log(log_path),
    )


def auto_refresh_dashboard(model_name: str, raw_dir: str, train_dir: str):
    try:
        return refresh_text_dashboard(model_name, raw_dir, train_dir)
    except Exception as exc:
        log_path = ACTIVE_TASK["log_path"]
        safe_model_name = sanitize_model_name(model_name)
        safe_raw_dir = sanitize_dataset_name(raw_dir) or safe_model_name
        safe_train_dir = train_dir or default_train_dir_for_dataset(safe_raw_dir)
        fallback_message = f"自动刷新异常：{type(exc).__name__}: {exc}"
        return (
            render_model_workspace_summary(safe_model_name),
            gr.update(label=dataset_file_list_label(safe_raw_dir)),
            render_dataset_file_list(safe_raw_dir),
            '<div class="stage-check-row"><div class="stage-check-title"><span class="stage-dot" style="color:#c0392b;">●</span>自动刷新失败，请稍后重试或手动点击刷新任务状态。</div></div>',
            render_preflight_check(safe_model_name, safe_raw_dir, safe_train_dir),
            render_stage_alert(safe_model_name, safe_raw_dir, safe_train_dir),
            "当前还没有自动 batch size 结果。",
            fallback_message,
            fallback_message,
            extract_log_highlights(log_path),
            tail_log(log_path),
        )


def set_active_task(task_name: str, cmd: Optional[List[str]], log_path: Path, proc=None, pipeline_name: Optional[str] = None):
    ACTIVE_TASK["name"] = task_name
    ACTIVE_TASK["proc"] = proc
    ACTIVE_TASK["log_path"] = log_path
    ACTIVE_TASK["cmd"] = cmd
    ACTIVE_TASK["stage_label"] = TASK_STAGE_LABELS.get(task_name, task_name)
    ACTIVE_TASK["pipeline_name"] = pipeline_name
    if ACTIVE_TASK["started_at"] is None:
        ACTIVE_TASK["started_at"] = time.time()


def start_pipeline(pipeline_name: str, steps: List[Tuple[str, List[str], bool]]):
    """按顺序执行一组子进程步骤，并把它们当作一个流程任务来跟踪。"""
    return start_pipeline_task(
        pipeline_name,
        steps,
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        pipeline_labels=PIPELINE_LABELS,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
        append_pipeline_log_fn=append_pipeline_log,
    )


def start_task(task_name: str, cmd: list[str]):
    return start_task_process(
        task_name,
        cmd,
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
    )


def stop_task():
    return stop_task_process(
        active_task=ACTIVE_TASK,
        current_stage_label_fn=current_stage_label,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
    )


def launch_resample(raw_dir: str, train_dir: str):
    """启动第 1 步：把原始 wav 整理到训练目录。"""
    return launch_resample_task(
        raw_dir,
        train_dir,
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
    )


def launch_config(model_name: str, train_dir: str, speech_encoder: str):
    """启动第 2 步：重新生成训练列表和配置文件。"""
    return launch_config_task(
        model_name,
        train_dir,
        normalize_speech_encoder_choice(speech_encoder),
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
    )


def launch_preprocess(model_name: str, train_dir: str):
    """启动第 3 步：提取内容、音高和频谱等特征。"""
    return launch_preprocess_task(
        model_name,
        train_dir,
        default_preprocess_workers=DEFAULT_PREPROCESS_WORKERS,
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
    )


def launch_train(model_name: str, auto_batch_probe: bool = False):
    """启动第 4 步：主模型训练。"""
    return launch_train_task(
        model_name,
        auto_batch_probe=auto_batch_probe,
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
    )


def launch_train_diff(model_name: str):
    """启动第 5 步：扩散训练。"""
    return launch_train_diff_task(
        model_name,
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
    )


def launch_train_index(model_name: str, train_dir: str):
    """启动第 6 步：训练检索增强用的特征索引。"""
    return launch_train_index_task(
        model_name,
        train_dir,
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
    )


def launch_pipeline_prep(model_name: str, raw_dir: str, train_dir: str, speech_encoder: str):
    """启动常用准备流程：重采样 -> 配置与列表 -> 提取特征。"""
    return launch_pipeline_prep_task(
        model_name,
        raw_dir,
        train_dir,
        normalize_speech_encoder_choice(speech_encoder),
        default_preprocess_workers=DEFAULT_PREPROCESS_WORKERS,
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        pipeline_labels=PIPELINE_LABELS,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
        append_pipeline_log_fn=append_pipeline_log,
    )


def launch_pipeline_train_main(model_name: str, raw_dir: str, train_dir: str, speech_encoder: str, auto_batch_probe: bool = False):
    """启动到主模型训练为止的完整流程。"""
    return launch_pipeline_train_main_task(
        model_name,
        raw_dir,
        train_dir,
        normalize_speech_encoder_choice(speech_encoder),
        auto_batch_probe=auto_batch_probe,
        default_preprocess_workers=DEFAULT_PREPROCESS_WORKERS,
        active_task=ACTIVE_TASK,
        task_log_dir=TASK_LOG_DIR,
        pipeline_labels=PIPELINE_LABELS,
        task_stage_labels=TASK_STAGE_LABELS,
        set_active_task_fn=set_active_task,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
        append_pipeline_log_fn=append_pipeline_log,
    )


def launch_tensorboard():
    tb_cmd = [sys.executable, "-m", "tensorboard.main", "--logdir=logs", "--port=6006"]
    current = ACTIVE_TASK["proc"]
    if current is not None and current.poll() is None:
        return "当前已有训练任务在运行。TensorBoard 不占用训练槽位，请手动在终端启动。"
    subprocess.Popen(tb_cmd, cwd=ROOT, start_new_session=True)
    opened = open_local_url("http://127.0.0.1:6006")
    if opened:
        message = "已启动并尝试打开训练监控：http://127.0.0.1:6006"
    else:
        message = "训练监控已启动，但浏览器没有成功打开。请手动访问：http://127.0.0.1:6006"
    set_ui_notice(message)
    return message


def open_local_url(url: str) -> bool:
    """尽量在当前系统里拉起本机浏览器打开指定地址。"""
    try:
        if webbrowser.open(url):
            return True
    except Exception:
        pass

    system_name = platform.system()
    try:
        if system_name == "Darwin":
            subprocess.Popen(["/usr/bin/open", url], cwd=ROOT, start_new_session=True)
            return True
        if system_name == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        if system_name == "Linux":
            subprocess.Popen(["/usr/bin/xdg-open", url], cwd=ROOT, start_new_session=True)
            return True
    except Exception:
        return False
    return False


def set_ui_notice(message: str, ttl_seconds: int = 15):
    """保存一条短时可见的页面提示，避免被定时刷新立即覆盖。"""
    UI_NOTICE["message"] = message
    UI_NOTICE["expires_at"] = time.time() + ttl_seconds


def render_log_highlights(path: Path):
    """优先显示短时页面提示，其次回退到日志高亮。"""
    expires_at = UI_NOTICE.get("expires_at", 0.0) or 0.0
    message = UI_NOTICE.get("message")
    if message and time.time() < expires_at:
        return message
    if message:
        UI_NOTICE["message"] = None
        UI_NOTICE["expires_at"] = 0.0
    return extract_log_highlights(path)


def launch_infer_ui():
    """在当前 Python 环境中打开或拉起推理页面。

    这里会记录当前训练页会话启动过的推理进程，避免误打开别的旧端口服务，
    特别是升级到 Gradio 4 后，环境不一致时这个问题会更明显。
    """
    try:
        existing_proc = INFER_UI_STATE["proc"]
        existing_port = INFER_UI_STATE["port"]
        if existing_proc is not None and existing_proc.poll() is None and existing_port:
            infer_url = f"http://127.0.0.1:{existing_port}"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", existing_port)) == 0:
                    if open_local_url(infer_url):
                        message = f"已打开当前会话的推理页面：{infer_url}"
                    else:
                        message = f"推理页面已就绪，但浏览器没有成功打开。请手动访问：{infer_url}"
                    set_ui_notice(message)
                    return message

        env = os.environ.copy()
        env["OPEN_BROWSER"] = "0"
        infer_port = find_available_port(7860)
        infer_url = f"http://127.0.0.1:{infer_port}"
        env["GRADIO_SERVER_PORT"] = str(infer_port)
        env["GRADIO_ANALYTICS_ENABLED"] = "False"
        proc = subprocess.Popen(
            [sys.executable, "app_infer.py"],
            cwd=ROOT,
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(20):
            if proc.poll() is not None:
                return f"推理页面启动失败：app_infer.py 已退出（退出码 {proc.returncode}）。请先在终端单独运行并查看报错。"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", infer_port)) == 0:
                    INFER_UI_STATE["proc"] = proc
                    INFER_UI_STATE["port"] = infer_port
                    if open_local_url(infer_url):
                        message = f"已启动并打开推理页面：{infer_url}"
                    else:
                        message = f"推理页面已启动，但浏览器没有成功打开。请手动访问：{infer_url}"
                    set_ui_notice(message)
                    return message
            time.sleep(0.25)

        INFER_UI_STATE["proc"] = proc
        INFER_UI_STATE["port"] = infer_port
        message = f"推理页面正在启动中：{infer_url}；如果浏览器没有自动打开，可手动访问这个地址。"
        set_ui_notice(message)
        return message
    except Exception as exc:
        message = f"打开推理页面失败：{type(exc).__name__}: {exc}"
        set_ui_notice(message)
        return message


def find_available_port(default_port: int, max_tries: int = 20):
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


def ensure_localhost_bypass_proxy():
    """确保本地回环地址不会误走系统代理，避免 Gradio 本地可达性探测失败。"""
    bypass_hosts = ["127.0.0.1", "localhost"]
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        entries = [item.strip() for item in current.split(",") if item.strip()]
        changed = False
        for host in bypass_hosts:
            if host not in entries:
                entries.append(host)
                changed = True
        if changed or not current:
            os.environ[key] = ",".join(entries)


with gr.Blocks(
    analytics_enabled=False,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
        font=["Avenir Next", "PingFang SC", "Helvetica Neue", "sans-serif"],
    ),
    css=TRAIN_PAGE_CSS,
) as app:
    gr.HTML("""
        <div class="hero">
            <h1>训练控制台</h1>
            <p>先补齐训练前依赖与底模，再确定当前模型工作区，最后导入数据并继续训练。</p>
        </div>
    """)
    initial_models = scan_model_workspaces()
    remembered_model = load_last_selected_model()
    initial_workspace_model = remembered_model if remembered_model in initial_models else (initial_models[0] if initial_models else "default_model")
    initial_workspace = load_model_workspace(initial_workspace_model)
    if initial_workspace is None:
        initial_workspace = save_model_workspace(initial_workspace_model, infer_workspace_dataset_name(initial_workspace_model))
    save_last_selected_model(initial_workspace_model)
    initial_dataset = infer_workspace_dataset_name(initial_workspace_model)
    initial_train_dir = default_train_dir_for_dataset(initial_dataset)
    initial_model_name = initial_workspace["model_name"]
    initial_model_name_sync_token = initial_model_name
    stage_alert = gr.Textbox(
        value=render_stage_alert(initial_model_name, initial_dataset, initial_train_dir),
        lines=1,
        interactive=False,
        show_label=False,
        elem_classes=["stage-alert-box"],
    )

    with gr.Row():
        with gr.Column(scale=12, elem_classes=["section-card"]):
            gr.Markdown("### 训练前依赖与底模")
            with gr.Row():
                with gr.Column(scale=6):
                    pretrain_status = gr.HTML(
                        value=render_pretrain_status(),
                    )
                with gr.Column(scale=6):
                    pretrain_progress = gr.HTML(render_pretrain_progress())
                    pretrain_asset_selector = gr.Dropdown(
                        label="选择依赖",
                        choices=pretrain_asset_choices(PRETRAIN_ASSETS, get_rmvpe_path(), is_rmvpe_asset_valid),
                        value=first_missing_pretrain_asset(PRETRAIN_ASSETS, get_rmvpe_path(), is_rmvpe_asset_valid) or "contentvec_hf",
                    )
                    pretrain_asset_guide = gr.HTML(render_pretrain_asset_guide("contentvec_hf"))
                    pretrain_asset_file = gr.File(
                        label="选择本地文件",
                        type="filepath",
                        file_types=PRETRAIN_ASSETS["contentvec_hf"].get("file_types"),
                    )
                    with gr.Row():
                        pretrain_asset_download_btn = gr.Button(
                            "自动获取当前依赖",
                            elem_classes=["info-action"],
                            elem_id="train-download-asset",
                        )
                        pretrain_asset_refresh_btn = gr.Button("刷新依赖状态", elem_classes=["info-action"], elem_id="train-refresh-assets")

    with gr.Row():
        with gr.Column(scale=5, elem_classes=["section-card"]):
            gr.Markdown("### 模型工作区")
            gr.Markdown("这里决定你当前在训练哪一个模型。新建模型后会进入新的工作区；在“切换模型”里选中后会立即切过去。")
            with gr.Row():
                workspace_new_name = gr.Textbox(label="新建模型名", value=initial_model_name, placeholder="例如：paimeng")
                workspace_selector = gr.Dropdown(label="切换模型", choices=initial_models, value=initial_model_name)
            with gr.Row():
                workspace_create_btn = gr.Button("新建训练模型", elem_classes=["primary-action"])
                workspace_delete_btn = gr.Button("删除当前模型工作区", elem_classes=["danger-action"], elem_id="train-delete-workspace")
            workspace_summary = gr.HTML(render_model_workspace_summary(initial_model_name))
            with gr.Group(visible=False, elem_id="action-block-modal") as action_block_dialog:
                with gr.Column(elem_id="action-block-card"):
                    gr.HTML('<div class="delete-modal-section"><h3>暂时不能操作</h3><p class="delete-modal-copy">请先处理当前任务，再继续这个操作。</p></div>')
                    with gr.Column(elem_classes=["delete-modal-body"]):
                        action_block_message = gr.Textbox(value="", lines=3, interactive=False, show_label=False)
                    with gr.Row(elem_id="action-block-actions"):
                        action_block_ok_btn = gr.Button("我知道了", elem_id="action-block-ok")
            gr.Markdown("#### 训练语音数据")
            train_model_name = gr.Textbox(value=initial_model_name, visible=False)
            model_name_sync_token = gr.Textbox(value=initial_model_name_sync_token, visible=False)
            dataset_source_dir = gr.Textbox(value=initial_dataset, visible=False)
            dataset_train_dir = gr.Textbox(value=initial_train_dir, visible=False)
            dataset_new_dir_name = gr.Textbox(value=initial_dataset, visible=False)
            with gr.Accordion(dataset_file_list_label(initial_dataset), open=False) as dataset_file_list_section:
                dataset_file_list = gr.HTML(
                value=render_dataset_file_list(initial_dataset),
                )
            delete_target_name = gr.Textbox(value="", visible=False)
            delete_action_kind = gr.Textbox(value="", visible=False)
            with gr.Group(visible=False, elem_id="delete-confirm-modal") as delete_confirm_row:
                with gr.Column(elem_id="delete-confirm-card"):
                    gr.HTML('<div class="delete-modal-section"><h3>删除确认</h3><p class="delete-modal-copy">这是一个不可恢复的删除操作，请确认后再继续。</p></div>')
                    with gr.Column(elem_classes=["delete-modal-body"]):
                        delete_confirm_message = gr.Textbox(value="", lines=6, interactive=False, show_label=False)
                    with gr.Row(elem_id="delete-confirm-actions"):
                        cancel_delete_btn = gr.Button("取消", elem_id="delete-cancel-action")
                        confirm_delete_btn = gr.Button("确认删除", elem_id="delete-confirm-action")
            dataset_import_dir = gr.File(label="为当前模型导入数据文件夹（内部直接是 .wav）", file_count="directory", type="filepath", elem_id="speaker-folder-upload")
            with gr.Row():
                refresh_overview_btn = gr.Button("刷新语音数据", elem_classes=["info-action"], elem_id="train-refresh-overview")
                delete_dataset_btn = gr.Button("删除语音数据", elem_classes=["danger-action"], elem_id="train-delete-dataset")
            dataset_import_message = gr.HTML(value=render_dataset_import_status_for_dataset(initial_dataset))

        with gr.Column(scale=7, elem_classes=["section-card"]):
            gr.Markdown("### 开始前检查")
            preflight_check = gr.HTML(render_preflight_check(initial_model_name, initial_dataset, initial_train_dir))
            gr.Markdown("### 训练步骤")
            speech_encoder_selector = gr.Textbox(
                value=load_model_speech_encoder(initial_model_name),
                visible=False,
            )
            task_refresh_token = gr.Textbox(value="0", visible=False)
            gr.Markdown("#### 当前进度")
            stage_judgement = gr.HTML(render_stage_judgement(initial_model_name, initial_dataset, initial_train_dir))
            runtime_banner = gr.Textbox(
                label="运行提示",
                value=render_runtime_banner(),
                lines=2,
                interactive=False,
                elem_classes=["runtime-banner-box"],
            )
            with gr.Row():
                resample_btn = gr.Button("1. 重采样到 dataset/44k", elem_classes=["primary-action"])
                config_btn = gr.Button("2. 生成配置与文件列表", elem_classes=["primary-action"])
            with gr.Row():
                preprocess_btn = gr.Button("3. 提取特征", elem_classes=["primary-action"])
                pipeline_prep_btn = gr.Button("一键执行 1-3 步", elem_classes=["primary-action"])
            with gr.Row():
                train_btn = gr.Button("4. 启动主模型训练", elem_classes=["primary-action"])
                pipeline_train_btn = gr.Button("一键执行到主模型训练", elem_classes=["primary-action"])
            auto_batch_probe_toggle = gr.Checkbox(
                label="训练前自动探测 batch size（实验项）",
                value=False,
                info="勾选后会先做单卡探测，再按安全余量折算成更适合日常使用的推荐 batch size 用于正式训练。",
            )
            auto_batch_probe_summary = gr.Textbox(
                label="自动 batch size 结果",
                value=render_auto_batch_probe_summary(ACTIVE_TASK["log_path"]),
                lines=2,
                interactive=False,
            )
            gr.Markdown("### 进阶训练")
            with gr.Row():
                train_diff_btn = gr.Button("5. 启动扩散训练", elem_classes=["secondary-action"])
                train_index_btn = gr.Button("6. 训练音色增强索引", elem_classes=["secondary-action"])
            gr.Markdown("### 运行与工具")
            with gr.Row():
                refresh_task_btn = gr.Button("更新任务状态", elem_classes=["info-action"], elem_id="train-refresh-task")
                stop_btn = gr.Button("停止当前任务", elem_classes=["danger-secondary"], elem_id="train-stop-task")
            with gr.Row():
                tensorboard_btn = gr.Button("打开训练监控", elem_classes=["info-action"], elem_id="train-open-tensorboard")
                open_infer_btn = gr.Button("进入推理界面", elem_classes=["info-action"], elem_id="train-open-infer")
    task_message = gr.Textbox(value=current_task_feedback(), visible=False)
    task_status = gr.Textbox(value=task_runtime_text(), visible=False)

    with gr.Row():
        with gr.Column(scale=12, elem_classes=["section-card"]):
            gr.Markdown("### 运行日志")
            task_panel_timer = gr.Timer(value=1, active=True)
            task_log_highlights = gr.Textbox(label="错误与操作提示", value=render_log_highlights(ACTIVE_TASK["log_path"]), lines=6)
            with gr.Accordion("查看运行日志", open=False):
                task_log = gr.Textbox(label="最近日志", value="日志尚不存在。", lines=12)

    refresh_outputs = [
        workspace_summary,
        dataset_file_list_section,
        dataset_file_list,
        stage_judgement,
        preflight_check,
        stage_alert,
        runtime_banner,
        auto_batch_probe_summary,
        task_message,
        task_status,
        task_log_highlights,
        task_log,
        pipeline_prep_btn,
        pipeline_train_btn,
        resample_btn,
        config_btn,
        preprocess_btn,
        train_btn,
        train_diff_btn,
        train_index_btn,
        workspace_selector,
        workspace_new_name,
        workspace_create_btn,
        workspace_delete_btn,
    ]
    auto_refresh_outputs = [
        workspace_summary,
        dataset_file_list_section,
        dataset_file_list,
        stage_judgement,
        preflight_check,
        stage_alert,
        auto_batch_probe_summary,
        task_message,
        task_status,
        task_log_highlights,
        task_log,
    ]

    refresh_overview_btn.click(refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False)
    refresh_task_btn.click(refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False)
    task_panel_timer.tick(
        refresh_live_task_panel,
        [train_model_name, dataset_source_dir, dataset_train_dir],
        [
            stage_alert,
            runtime_banner,
            auto_batch_probe_summary,
            task_message,
            task_status,
            task_log_highlights,
            task_log,
            pipeline_prep_btn,
            pipeline_train_btn,
            resample_btn,
            config_btn,
            preprocess_btn,
            train_btn,
            train_diff_btn,
            train_index_btn,
            workspace_selector,
            workspace_new_name,
            workspace_create_btn,
            workspace_delete_btn,
            task_refresh_token,
        ],
        queue=False,
        show_api=False,
    )
    task_refresh_token.change(
        refresh_dashboard,
        [train_model_name, dataset_source_dir, dataset_train_dir],
        refresh_outputs,
        show_api=False,
    )
    workspace_create_btn.click(
        create_model_workspace,
        [workspace_new_name, dataset_source_dir],
        [workspace_selector, workspace_new_name, train_model_name, dataset_source_dir, dataset_train_dir, model_name_sync_token, speech_encoder_selector, workspace_summary, dataset_import_message],
        show_api=False,
    ).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    workspace_selector.change(
        switch_model_workspace,
        [workspace_selector, train_model_name, dataset_source_dir, dataset_train_dir, model_name_sync_token],
        [
            workspace_selector,
            workspace_new_name,
            train_model_name,
            dataset_source_dir,
            dataset_train_dir,
            model_name_sync_token,
            speech_encoder_selector,
            workspace_summary,
            dataset_import_message,
            action_block_message,
            action_block_dialog,
        ],
        show_api=False,
    ).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    workspace_delete_btn.click(
        prepare_delete_model_workspace,
        [workspace_selector],
        [
            delete_confirm_message,
            delete_confirm_row,
            delete_target_name,
            delete_action_kind,
            dataset_import_message,
            action_block_message,
            action_block_dialog,
        ],
        show_api=False,
    )
    action_block_ok_btn.click(
        lambda: (gr.update(value=""), gr.update(visible=False)),
        [],
        [action_block_message, action_block_dialog],
        show_api=False,
    )
    pretrain_asset_selector.change(
        lambda selected_asset: render_pretrain_asset_guide(selected_asset),
        [pretrain_asset_selector],
        [pretrain_asset_guide],
        show_api=False,
    ).then(
        pretrain_file_update,
        [pretrain_asset_selector],
        [pretrain_asset_file],
        show_api=False,
    )
    pretrain_asset_file.change(
        import_pretrain_asset,
        [pretrain_asset_selector, pretrain_asset_file],
        [pretrain_asset_guide, pretrain_status, pretrain_asset_file],
        show_api=False,
    ).then(
        pretrain_selector_update,
        [pretrain_asset_selector],
        [pretrain_asset_selector],
        show_api=False,
    ).then(
        render_pretrain_progress,
        [],
        [pretrain_progress],
        show_api=False,
    ).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    pretrain_asset_refresh_btn.click(
        lambda selected_asset: (
            render_pretrain_status(),
            render_pretrain_asset_guide(selected_asset),
        ),
        [pretrain_asset_selector],
        [pretrain_status, pretrain_asset_guide],
        show_api=False,
    ).then(
        pretrain_selector_update,
        [pretrain_asset_selector],
        [pretrain_asset_selector],
        show_api=False,
    ).then(
        render_pretrain_progress,
        [],
        [pretrain_progress],
        show_api=False,
    ).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    pretrain_asset_download_btn.click(
        download_pretrain_asset,
        [pretrain_asset_selector],
        [pretrain_asset_guide, pretrain_status],
        show_api=False,
    ).then(
        pretrain_selector_update,
        [pretrain_asset_selector],
        [pretrain_asset_selector],
        show_api=False,
    ).then(
        render_pretrain_progress,
        [],
        [pretrain_progress],
        show_api=False,
    ).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    dataset_import_dir.change(
        import_dataset_directory,
        [dataset_import_dir, dataset_source_dir],
        [dataset_import_message, dataset_source_dir],
        show_api=False,
    ).then(
        bind_workspace_dataset, [train_model_name, dataset_source_dir], [workspace_summary], show_api=False
    ).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    dataset_train_dir.change(refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False)
    delete_dataset_btn.click(
        prepare_delete_dataset,
        [dataset_source_dir],
        [delete_confirm_message, delete_confirm_row, delete_target_name, delete_action_kind, action_block_message, action_block_dialog],
        show_api=False,
    )
    confirm_delete_btn.click(
        confirm_delete_action,
        [delete_action_kind, delete_target_name],
        [
            dataset_import_message,
            dataset_source_dir,
            dataset_file_list,
            dataset_new_dir_name,
            dataset_train_dir,
            train_model_name,
            workspace_selector,
            workspace_new_name,
            model_name_sync_token,
            speech_encoder_selector,
            workspace_summary,
            delete_confirm_row,
            delete_target_name,
            delete_action_kind,
        ],
        show_api=False,
    ).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    cancel_delete_btn.click(
        lambda: (render_dataset_import_result("已取消删除。"), gr.update(visible=False), gr.update(value=""), gr.update(value="")),
        [],
        [dataset_import_message, delete_confirm_row, delete_target_name, delete_action_kind],
        show_api=False,
    )
    resample_btn.click(launch_resample, [dataset_source_dir, dataset_train_dir], [task_message, task_status, task_log], show_api=False).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    config_btn.click(launch_config, [train_model_name, dataset_train_dir, speech_encoder_selector], [task_message, task_status, task_log], show_api=False).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    preprocess_btn.click(launch_preprocess, [train_model_name, dataset_train_dir], [task_message, task_status, task_log], show_api=False).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    train_btn.click(launch_train, [train_model_name, auto_batch_probe_toggle], [task_message, task_status, task_log], show_api=False).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    train_diff_btn.click(launch_train_diff, [train_model_name], [task_message, task_status, task_log], show_api=False).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    train_index_btn.click(launch_train_index, [train_model_name, dataset_train_dir], [task_message, task_status, task_log], show_api=False).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    pipeline_prep_btn.click(launch_pipeline_prep, [train_model_name, dataset_source_dir, dataset_train_dir, speech_encoder_selector], [task_message, task_status, task_log], show_api=False).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    pipeline_train_btn.click(
        launch_pipeline_train_main,
        [train_model_name, dataset_source_dir, dataset_train_dir, speech_encoder_selector, auto_batch_probe_toggle],
        [task_message, task_status, task_log],
        show_api=False,
    ).then(refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False)
    stop_btn.click(stop_task, [], [task_message, task_status, task_log], show_api=False).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    tensorboard_btn.click(launch_tensorboard, [], [task_log_highlights], show_api=False)
    open_infer_btn.click(launch_infer_ui, [], [task_log_highlights], show_api=False)
    app.load(auto_refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], auto_refresh_outputs, show_api=False)

    ensure_localhost_bypass_proxy()
    server_port = find_available_port(7861)

    if os.environ.get("OPEN_BROWSER", "1") != "0":
        open_local_url(f"http://127.0.0.1:{server_port}")
    app.launch(server_name="127.0.0.1", server_port=server_port, share=False)


if __name__ == "__main__":
    pass
