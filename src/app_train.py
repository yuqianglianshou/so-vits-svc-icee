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
import ssl
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from huggingface_hub import hf_hub_download
from src.gradio_api_info_fallback import apply_gradio_4_api_info_patch
from src.path_utils import (
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
from src.train_ui.paths import (
    default_train_dir_for_dataset,
    model_config_path,
    sanitize_dataset_name,
    sanitize_model_name,
)
from src.train_ui.state import build_button_state_maps, collect_stage_state, render_stage_judgement_html
from src.train_ui.tasks import (
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
from src.train_ui.pretrain import (
    first_missing_pretrain_asset,
    is_pretrain_asset_ready,
    normalize_asset_key,
    ordered_pretrain_asset_keys,
    pretrain_asset_choices,
    render_pretrain_asset_guide as render_pretrain_asset_guide_html,
    render_pretrain_status as render_pretrain_status_html,
    resolve_uploaded_paths,
)
from src.train_ui.workspaces import (
    create_model_workspace_action,
    delete_dataset_directory,
    delete_model_workspace_action,
    dataset_file_list_label,
    infer_workspace_dataset_name,
    import_dataset_directory,
    load_last_selected_model,
    load_model_workspace,
    prepare_delete_dataset as prepare_delete_dataset_action,
    prepare_delete_model_workspace_action,
    render_dataset_file_list,
    render_dataset_import_status_for_dataset,
    render_model_workspace_summary,
    save_last_selected_model,
    save_model_workspace,
    scan_dataset_candidates,
    scan_model_workspaces,
    switch_model_workspace_action,
    suggest_next_dataset_name,
)
from src.train_ui.panels import build_stage_alert_text, build_task_feedback
from src.train_ui.runtime import (
    render_preflight_check as render_preflight_check_html,
    render_runtime_banner as render_runtime_banner_text,
    resolve_task_log_path,
    render_task_panel_snapshot as render_task_panel_snapshot_text,
    tail_log,
    task_runtime_text as task_runtime_text_render,
)
from src.train_ui.config_sync import (
    load_model_batch_size as load_model_batch_size_sync,
    load_template_batch_size as load_template_batch_size_sync,
    persist_batch_size as persist_batch_size_sync,
)
from src.train_ui.launchers import (
    ensure_localhost_bypass_proxy,
    find_available_port,
    launch_infer_ui as launch_infer_ui_action,
    launch_tensorboard as launch_tensorboard_action,
    open_local_url,
)
from src.train_ui.dashboard import (
    auto_refresh_dashboard as auto_refresh_dashboard_action,
    refresh_dashboard as refresh_dashboard_action,
    refresh_live_task_panel as refresh_live_task_panel_action,
    refresh_text_dashboard as refresh_text_dashboard_action,
)
from src.train_ui.text import (
    format_duration,
    render_dataset_import_result,
    render_pretrain_progress as render_pretrain_progress_html,
)
from src.utils import get_supported_speech_encoders

apply_gradio_4_api_info_patch()


CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent
TRAIN_PAGE_CSS = (CODE_ROOT / "train_ui" / "page.css").read_text(encoding="utf-8")
TASK_LOG_DIR = ROOT / "model_assets/workspaces" / "webui_tasks"
TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_TASK = {
    "name": None,
    "proc": None,
    "log_path": None,
    "display_log_path": None,
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

TENSORBOARD_STATE = {
    "proc": None,
    "port": 6006,
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
DEFAULT_BATCH_SIZE = json.loads((ROOT / "config_templates" / "config_template.json").read_text(encoding="utf-8"))["train"]["batch_size"]
CONFIG_TEMPLATE_PATH = ROOT / "config_templates" / "config_template.json"
DIFFUSION_TEMPLATE_PATH = ROOT / "config_templates" / "diffusion_template.yaml"

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

RAW_DATASET_PARENT = ROOT / "training_data/source"

PRETRAIN_ASSETS = {
    "contentvec_hf": {
        "label": "ContentVec HF 模型目录 contentvec_hf/",
        "target": ENCODER_DIR / "contentvec_hf",
        "accepted_names": {"contentvec_hf.zip", "config.json", "model.safetensors"},
        "file_types": [".zip", ".json", ".safetensors"],
        "required_files": ["config.json", "model.safetensors"],
        "download_links": [
            ("config.json 直链", "https://huggingface.co/lengyue233/content-vec-best/resolve/ab04aa7067b99ee05cc82499bc64916b980a1967/config.json"),
            ("model.safetensors 直链", "https://huggingface.co/lengyue233/content-vec-best/resolve/60a4eafc5775c9ff1f813fa544c0c8d3099898f2/model.safetensors"),
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


def is_pretrain_section_open() -> bool:
    return not all(
        is_pretrain_asset_ready(asset, get_rmvpe_path(), is_rmvpe_asset_valid)
        for asset in PRETRAIN_ASSETS.values()
    )


def refresh_pretrain_section_open():
    return gr.update(open=is_pretrain_section_open())


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
    source_path_list = [Path(path_str) for path_str in resolve_uploaded_paths(uploaded_file)]
    if not source_path_list:
        return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)
    if any(not source_path.exists() for source_path in source_path_list):
        return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)

    allowed_types = {suffix.lower() for suffix in asset.get("file_types", [])}
    if allowed_types and any(source_path.suffix.lower() not in allowed_types for source_path in source_path_list):
        allowed = "、".join(sorted(allowed_types))
        return (
            render_pretrain_asset_guide(asset_key),
            render_pretrain_status(),
            gr.update(value=None),
        )

    if any(source_path.name not in asset["accepted_names"] for source_path in source_path_list):
        accepted = "、".join(sorted(asset["accepted_names"]))
        return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)

    if asset_key == "contentvec_hf":
        target_dir = asset["target"]
        target_dir.mkdir(parents=True, exist_ok=True)
        if len(source_path_list) == 1 and source_path_list[0].suffix.lower() == ".zip":
            with zipfile.ZipFile(source_path_list[0], "r") as zf:
                zf.extractall(target_dir)
        else:
            for source_path in source_path_list:
                if source_path.name in {"config.json", "model.safetensors"}:
                    shutil.copyfile(source_path, target_dir / source_path.name)
        return render_pretrain_asset_guide(asset_key), render_pretrain_status(), gr.update(value=None)

    zip_targets = {
        "rmvpe": {"rmvpe.pt", "model.pt"},
        "sovits_g0": {"G_0.pth"},
        "sovits_d0": {"D_0.pth"},
        "diffusion_model_0": {"model_0.pt"},
    }
    source_path = source_path_list[0]

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


def render_workspace_refresh_updates(current_model_name: str):
    locked = has_active_task()
    safe_model_name = sanitize_model_name(current_model_name)
    choices = scan_model_workspaces()
    value = safe_model_name if safe_model_name in choices else (choices[0] if choices else "default_model")
    return (
        gr.update(choices=choices, value=value, interactive=not locked),
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


def create_model_workspace(new_model_name: str, current_dataset_name: str):
    return create_model_workspace_action(
        new_model_name,
        current_dataset_name,
        has_active_task_fn=has_active_task,
        active_task_block_message_fn=active_task_block_message,
        speech_encoder_value_update_fn=speech_encoder_value_update,
        load_model_speech_encoder_fn=load_model_speech_encoder,
    )


def switch_model_workspace(selected_model_name: str, current_model_name: str, current_dataset_name: str, current_train_dir: str, current_sync_token: str):
    return switch_model_workspace_action(
        selected_model_name,
        current_model_name,
        current_dataset_name,
        current_train_dir,
        current_sync_token,
        active_task_name=sanitize_model_name(ACTIVE_TASK.get("name") or ""),
        has_active_task_fn=has_active_task,
        speech_encoder_value_update_fn=speech_encoder_value_update,
        load_model_speech_encoder_fn=load_model_speech_encoder,
        active_task_block_message_fn=active_task_block_message,
        show_active_task_block_dialog_fn=show_active_task_block_dialog,
    )


def prepare_delete_model_workspace(selected_model_name: str):
    return prepare_delete_model_workspace_action(
        selected_model_name,
        has_active_task_fn=has_active_task,
        active_task_block_message_fn=active_task_block_message,
        show_active_task_block_dialog_fn=show_active_task_block_dialog,
    )


def delete_model_workspace(selected_model_name: str):
    return delete_model_workspace_action(
        selected_model_name,
        speech_encoder_value_update_fn=speech_encoder_value_update,
        load_model_speech_encoder_fn=load_model_speech_encoder,
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


def prepare_delete_dataset(dataset_name: str):
    if has_active_task():
        return (
            "",
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            *show_active_task_block_dialog("删除当前模型数据目录"),
        )
    return prepare_delete_dataset_action(dataset_name)


def refresh_dataset_name_suggestion():
    return gr.update(value=suggest_next_dataset_name())


def detect_file(path_str: str):
    path = ROOT / path_str
    return "已就绪" if path.exists() else "缺失"


def append_pipeline_log(log_path: Path, message: str):
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message.rstrip() + "\n")
def render_stage_judgement(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "training_data/processed/44k"):
    state = collect_stage_state(model_name, raw_dir, train_dir)
    return render_stage_judgement_html(state)


def render_button_updates(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "training_data/processed/44k"):
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


def current_task_feedback(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "training_data/processed/44k"):
    proc = ACTIVE_TASK["proc"]
    task_name = ACTIVE_TASK["name"]
    log_path = resolve_task_log_path(ACTIVE_TASK)
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


def render_stage_alert(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "training_data/processed/44k"):
    proc = ACTIVE_TASK["proc"]
    task_name = ACTIVE_TASK["name"]
    stage_label = current_stage_label()
    is_running = proc is not None and proc.poll() is None
    succeeded = proc is not None and proc.returncode == 0
    return build_stage_alert_text(task_name, stage_label, is_running, succeeded)


def render_preflight_check(model_name: str = "44k", raw_dir: str = "default_dataset", train_dir: str = "training_data/processed/44k"):
    return render_preflight_check_html(
        model_name,
        raw_dir,
        train_dir,
        get_sovits_g0_path_fn=get_sovits_g0_path,
        get_sovits_d0_path_fn=get_sovits_d0_path,
        get_diffusion_model_0_path_fn=get_diffusion_model_0_path,
        get_rmvpe_path_fn=get_rmvpe_path,
        is_rmvpe_asset_valid_fn=is_rmvpe_asset_valid,
        get_contentvec_hf_path_fn=get_contentvec_hf_path,
        get_nsf_hifigan_model_path_fn=get_nsf_hifigan_model_path,
        get_nsf_hifigan_config_path_fn=get_nsf_hifigan_config_path,
    )


def task_runtime_text():
    return task_runtime_text_render(
        ACTIVE_TASK,
        current_stage_label_fn=current_stage_label,
        pipeline_labels=PIPELINE_LABELS,
    )


def render_runtime_banner():
    return render_runtime_banner_text(
        ACTIVE_TASK,
        current_stage_label_fn=current_stage_label,
        pipeline_labels=PIPELINE_LABELS,
    )


def load_model_batch_size(model_name: str):
    return load_model_batch_size_sync(model_name, CONFIG_TEMPLATE_PATH, DEFAULT_BATCH_SIZE)


def load_template_batch_size() -> int:
    return load_template_batch_size_sync(CONFIG_TEMPLATE_PATH, DEFAULT_BATCH_SIZE)


def persist_batch_size(model_name: str, batch_size: int | float | None):
    return persist_batch_size_sync(
        model_name,
        batch_size,
        config_template_path=CONFIG_TEMPLATE_PATH,
        diffusion_template_path=DIFFUSION_TEMPLATE_PATH,
        default_batch_size=DEFAULT_BATCH_SIZE,
    )


def render_task_panel_snapshot(model_name: str, raw_dir: str, train_dir: str):
    return render_task_panel_snapshot_text(
        model_name,
        raw_dir,
        train_dir,
        active_task=ACTIVE_TASK,
        render_stage_alert_fn=render_stage_alert,
        current_task_feedback_fn=current_task_feedback,
        current_stage_label_fn=current_stage_label,
        pipeline_labels=PIPELINE_LABELS,
    )


def refresh_live_task_panel(model_name: str, raw_dir: str, train_dir: str):
    return refresh_live_task_panel_action(
        model_name,
        raw_dir,
        train_dir,
        active_task=ACTIVE_TASK,
        task_lifecycle_cache=TASK_LIFECYCLE_CACHE,
        render_button_updates_fn=render_button_updates,
        render_workspace_live_control_updates_fn=render_workspace_control_updates,
        render_task_panel_snapshot_fn=render_task_panel_snapshot,
    )


def refresh_dashboard(model_name: str, raw_dir: str, train_dir: str):
    return refresh_dashboard_action(
        model_name,
        raw_dir,
        train_dir,
        render_model_workspace_summary_fn=render_model_workspace_summary,
        dataset_file_list_label_fn=dataset_file_list_label,
        render_dataset_file_list_fn=render_dataset_file_list,
        render_stage_judgement_fn=render_stage_judgement,
        render_preflight_check_fn=render_preflight_check,
        render_task_panel_snapshot_fn=render_task_panel_snapshot,
        load_model_batch_size_fn=load_model_batch_size,
        render_button_updates_fn=render_button_updates,
        render_workspace_refresh_updates_fn=render_workspace_refresh_updates,
    )


def refresh_text_dashboard(model_name: str, raw_dir: str, train_dir: str):
    return refresh_text_dashboard_action(
        model_name,
        raw_dir,
        train_dir,
        active_task=ACTIVE_TASK,
        render_model_workspace_summary_fn=render_model_workspace_summary,
        dataset_file_list_label_fn=dataset_file_list_label,
        render_dataset_file_list_fn=render_dataset_file_list,
        render_stage_judgement_fn=render_stage_judgement,
        render_preflight_check_fn=render_preflight_check,
        render_stage_alert_fn=render_stage_alert,
        current_task_feedback_fn=current_task_feedback,
        task_runtime_text_fn=task_runtime_text,
        tail_log_fn=tail_log,
        resolve_task_log_path_fn=resolve_task_log_path,
    )


def auto_refresh_dashboard(model_name: str, raw_dir: str, train_dir: str):
    return auto_refresh_dashboard_action(
        model_name,
        raw_dir,
        train_dir,
        active_task=ACTIVE_TASK,
        refresh_text_dashboard_fn=refresh_text_dashboard,
        render_model_workspace_summary_fn=render_model_workspace_summary,
        dataset_file_list_label_fn=dataset_file_list_label,
        render_dataset_file_list_fn=render_dataset_file_list,
        render_preflight_check_fn=render_preflight_check,
        render_stage_alert_fn=render_stage_alert,
        tail_log_fn=tail_log,
        resolve_task_log_path_fn=resolve_task_log_path,
    )


def set_active_task(task_name: str, cmd: Optional[List[str]], log_path: Path, proc=None, pipeline_name: Optional[str] = None, display_log_path: Optional[Path] = None):
    ACTIVE_TASK["name"] = task_name
    ACTIVE_TASK["proc"] = proc
    ACTIVE_TASK["log_path"] = log_path
    ACTIVE_TASK["display_log_path"] = display_log_path
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


def launch_train(model_name: str, batch_size: int | float | None = None):
    """启动第 4 步：主模型训练。"""
    return launch_train_task(
        model_name,
        batch_size=batch_size,
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


def launch_pipeline_train_main(model_name: str, raw_dir: str, train_dir: str, speech_encoder: str, batch_size: int | float | None = None):
    """启动到主模型训练为止的完整流程。"""
    return launch_pipeline_train_main_task(
        model_name,
        raw_dir,
        train_dir,
        normalize_speech_encoder_choice(speech_encoder),
        batch_size=batch_size,
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
    return launch_tensorboard_action(root=ROOT, tensorboard_state=TENSORBOARD_STATE, ui_notice=UI_NOTICE)


def launch_infer_ui():
    return launch_infer_ui_action(root=ROOT, infer_ui_state=INFER_UI_STATE, ui_notice=UI_NOTICE)


def launch_tensorboard_feedback():
    message = launch_tensorboard()
    return message, message


def launch_infer_ui_feedback():
    message = launch_infer_ui()
    return message, message


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
            with gr.Accordion("查看与处理训练前依赖", open=is_pretrain_section_open()) as pretrain_section:
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
                            file_count="multiple",
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
                resample_btn = gr.Button("1. 重采样到 training_data/processed/44k", elem_classes=["primary-action"])
                config_btn = gr.Button("2. 生成配置与文件列表", elem_classes=["primary-action"])
            with gr.Row():
                preprocess_btn = gr.Button("3. 提取特征", elem_classes=["primary-action"])
                pipeline_prep_btn = gr.Button("一键执行 1-3 步", elem_classes=["primary-action"])
            gr.Markdown(
                "#### Batch Size 说明\n"
                "- `batch size` 可以理解为：**一次送进 GPU 同时训练的样本量**。 当前项目默认 `8`。\n"
                "- 一般来说，`batch size` 越大，训练吞吐越高；但太大容易把显存和整机资源吃满，Windows 桌面也会明显变卡。太小影响训练效率，表现为 训练模型所用时间长。\n"
                "- 实用经验是：**GPU 利用率大致在 80% 到接近 100% 波动**，往往已经是比较合适的区间，这样训练过程中还可以开个网页看个视频，再高的话就明显卡顿了。\n"
                "- `8GB` 显卡建议选择 `8`，`12GB` 显卡建议先调整到 `12`；开启训练运行几分钟后，打开资源管理器，查看当前GPU使用率，小于 80% 时，点击 停止当前任务 ，将此值加2，再次开启训练，看GPU使用率。一直在100%时，点击 停止当前任务 ，将此值减2，再看GPU使用率。调整到合适为止。\n"
            )
            train_batch_size_value = gr.Number(
                label="当前训练使用 batch size（可改）",
                value=load_model_batch_size(initial_model_name),
                precision=0,
                minimum=1,
                info="主模型训练实际会使用这个值。修改后会立即同步到当前模型配置和模板配置。建议先从较保守的值开始，训练稳定后再逐步上调。",
            )
            with gr.Row():
                train_btn = gr.Button("4. 启动主模型训练", elem_classes=["primary-action"])
            with gr.Row():
                pipeline_train_btn = gr.Button("一键执行到主模型训练", elem_classes=["primary-action"])
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
            task_log = gr.Textbox(label="最近日志", value="日志尚不存在。", lines=16)

    refresh_outputs = [
        workspace_summary,
        dataset_file_list_section,
        dataset_file_list,
        stage_judgement,
        preflight_check,
        stage_alert,
        runtime_banner,
        train_batch_size_value,
        task_message,
        task_status,
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
        task_message,
        task_status,
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
            task_message,
            task_status,
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
        refresh_pretrain_section_open,
        [],
        [pretrain_section],
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
        refresh_pretrain_section_open,
        [],
        [pretrain_section],
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
        refresh_pretrain_section_open,
        [],
        [pretrain_section],
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
    train_batch_size_value.change(
        persist_batch_size,
        [train_model_name, train_batch_size_value],
        [train_batch_size_value, task_message],
        show_api=False,
    )
    train_btn.click(launch_train, [train_model_name, train_batch_size_value], [task_message, task_status, task_log], show_api=False).then(
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
        [train_model_name, dataset_source_dir, dataset_train_dir, speech_encoder_selector, train_batch_size_value],
        [task_message, task_status, task_log],
        show_api=False,
    ).then(refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False)
    stop_btn.click(stop_task, [], [task_message, task_status, task_log], show_api=False).then(
        refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], refresh_outputs, show_api=False
    )
    tensorboard_btn.click(launch_tensorboard_feedback, [], [task_message, runtime_banner], show_api=False)
    open_infer_btn.click(launch_infer_ui_feedback, [], [task_message, runtime_banner], show_api=False)
    app.load(auto_refresh_dashboard, [train_model_name, dataset_source_dir, dataset_train_dir], auto_refresh_outputs, show_api=False)

    ensure_localhost_bypass_proxy()
    server_port = find_available_port(7861)

    if os.environ.get("OPEN_BROWSER", "1") != "0":
        open_local_url(f"http://127.0.0.1:{server_port}", root=ROOT)
    app.launch(server_name="127.0.0.1", server_port=server_port, share=False)


if __name__ == "__main__":
    pass
