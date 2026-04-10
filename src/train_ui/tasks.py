"""训练页任务启动、停止与一键流程编排。

这个模块负责训练页面里真正会启动子进程的那一层逻辑，包括：
1. 单步任务启动，
2. 一键流程编排，
3. 当前任务停止。

页面入口只需要把当前状态对象和少量回调传进来，就能复用这些能力。
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from src.path_utils import ensure_runtime_base_models
from src.train_ui.paths import (
    ROOT,
    default_train_dir_for_dataset,
    model_config_path,
    model_diff_config_path,
    model_diffusion_dir,
    model_root_dir,
    model_train_list_path,
    model_val_list_path,
    resolve_raw_dataset_dir,
    sanitize_dataset_name,
    sanitize_model_name,
)
from src.train_ui.workspace import count_raw_dataset_wavs

WINDOWS_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def _find_available_local_port(default_port: int, max_tries: int = 50) -> int:
    for offset in range(max_tries):
        port = default_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"Unable to find an available port in {default_port}-{default_port + max_tries - 1}.")


def _assign_train_port(model_name: str, default_port: int = 8001) -> int:
    config_path = model_config_path(model_name)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    chosen_port = _find_available_local_port(default_port)
    config.setdefault("train", {})["port"] = str(chosen_port)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return chosen_port


def _task_running(active_task: dict):
    proc = active_task["proc"]
    thread = active_task["thread"]
    return (proc is not None and proc.poll() is None) or (thread is not None and thread.is_alive())


def _spawn_popen(cmd: list[str], log_file):
    kwargs = {
        "cwd": ROOT,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        kwargs["creationflags"] = WINDOWS_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def _terminate_process_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=3)
            return
        except Exception:
            pass
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()


def _running_conflict_response(active_task: dict, task_runtime_text_fn: Callable[[], str], tail_log_fn: Callable[[Path], str]):
    log_path = active_task.get("display_log_path") or active_task["log_path"]
    return (
        f"已有任务正在运行：{active_task['pipeline_name'] or active_task['name']}。\n请先停止当前任务，再启动新的任务。",
        task_runtime_text_fn(),
        tail_log_fn(log_path),
    )


def start_pipeline(
    pipeline_name: str,
    steps,
    *,
    active_task: dict,
    task_log_dir: Path,
    pipeline_labels: dict,
    task_stage_labels: dict,
    set_active_task_fn: Callable,
    task_runtime_text_fn: Callable[[], str],
    tail_log_fn: Callable[[Path], str],
    append_pipeline_log_fn: Callable[[Path, str], None],
):
    """按顺序执行一组子进程步骤，并把它们当作一个流程任务来跟踪。"""
    if _task_running(active_task):
        return _running_conflict_response(active_task, task_runtime_text_fn, tail_log_fn)

    log_path = task_log_dir / f"{pipeline_name}_{int(time.time())}.log"
    active_task["started_at"] = time.time()
    active_task["stop_requested"] = False
    active_task["thread"] = None
    set_active_task_fn(steps[0][0], steps[0][1], log_path, proc=None, pipeline_name=pipeline_name, display_log_path=None)
    active_task["stage_label"] = "准备开始"

    def runner():
        try:
            append_pipeline_log_fn(log_path, f"[Pipeline] {pipeline_labels[pipeline_name]} 已启动")
            for task_name, cmd, wait_for_exit in steps:
                if active_task["stop_requested"]:
                    append_pipeline_log_fn(log_path, "[Pipeline] 收到停止信号，流程已终止。")
                    active_task["stage_label"] = "流程已停止"
                    active_task["proc"] = None
                    return

                append_pipeline_log_fn(log_path, f"[Pipeline] 开始 {task_stage_labels.get(task_name, task_name)}")
                log_file = log_path.open("a", encoding="utf-8")
                proc = _spawn_popen(cmd, log_file)
                set_active_task_fn(task_name, cmd, log_path, proc=proc, pipeline_name=pipeline_name, display_log_path=None)

                if not wait_for_exit:
                    append_pipeline_log_fn(log_path, f"[Pipeline] 已自动切换到 {task_stage_labels.get(task_name, task_name)}。")
                    return

                return_code = proc.wait()
                if return_code != 0:
                    append_pipeline_log_fn(log_path, f"[Pipeline] {task_stage_labels.get(task_name, task_name)} 失败，退出码 {return_code}。")
                    active_task["stage_label"] = f"{task_stage_labels.get(task_name, task_name)} 失败"
                    return

                append_pipeline_log_fn(log_path, f"[Pipeline] {task_stage_labels.get(task_name, task_name)} 完成。")
                active_task["proc"] = None

            active_task["stage_label"] = "流程已完成"
            active_task["name"] = pipeline_name
            active_task["cmd"] = None
            append_pipeline_log_fn(log_path, "[Pipeline] 全部步骤已完成。")
        finally:
            active_task["thread"] = None

    pipeline_thread = threading.Thread(target=runner, daemon=True)
    active_task["thread"] = pipeline_thread
    pipeline_thread.start()
    return (
        f"已启动流程：{pipeline_labels[pipeline_name]}\n日志文件：{log_path}",
        task_runtime_text_fn(),
        tail_log_fn(log_path),
    )


def start_task(
    task_name: str,
    cmd: list[str],
    *,
    active_task: dict,
    task_log_dir: Path,
    task_stage_labels: dict,
    set_active_task_fn: Callable,
    task_runtime_text_fn: Callable[[], str],
    tail_log_fn: Callable[[Path], str],
    display_log_path: Path | None = None,
):
    """启动单个训练子任务，并接管当前活动任务状态。"""
    if _task_running(active_task):
        return _running_conflict_response(active_task, task_runtime_text_fn, tail_log_fn)

    log_path = task_log_dir / f"{task_name}_{int(time.time())}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = _spawn_popen(cmd, log_file)
    active_task["thread"] = None
    active_task["pipeline_name"] = None
    active_task["stop_requested"] = False
    active_task["started_at"] = time.time()
    set_active_task_fn(task_name, cmd, log_path, proc=proc, pipeline_name=None, display_log_path=display_log_path)
    return (
        f"已启动任务：{task_stage_labels.get(task_name, task_name)}\n日志文件：{log_path}",
        task_runtime_text_fn(),
        tail_log_fn(display_log_path or log_path),
    )


def stop_task(
    *,
    active_task: dict,
    current_stage_label_fn: Callable[[], str],
    task_runtime_text_fn: Callable[[], str],
    tail_log_fn: Callable[[Path], str],
):
    """向当前任务发送停止信号。"""
    proc = active_task["proc"]
    thread = active_task["thread"]
    log_path = active_task.get("display_log_path") or active_task["log_path"]
    if (proc is None or proc.poll() is not None) and not (thread is not None and thread.is_alive()):
        return "当前没有可停止的任务。", task_runtime_text_fn(), tail_log_fn(log_path)
    active_task["stop_requested"] = True
    _terminate_process_tree(proc)
    return f"已发送停止信号：{active_task['pipeline_name'] or current_stage_label_fn()}", task_runtime_text_fn(), tail_log_fn(log_path)


def launch_resample(raw_dir: str, train_dir: str, *, active_task: dict, task_log_dir: Path, task_stage_labels: dict, set_active_task_fn: Callable, task_runtime_text_fn: Callable[[], str], tail_log_fn: Callable[[Path], str]):
    """启动第 1 步：把原始 wav 整理到训练目录。"""
    return start_task(
        "resample",
        [sys.executable, "-m", "src.train_pipeline.resample", "--in_dir", "training_data/source", "--speaker", raw_dir, "--out_dir2", train_dir],
        active_task=active_task,
        task_log_dir=task_log_dir,
        task_stage_labels=task_stage_labels,
        set_active_task_fn=set_active_task_fn,
        task_runtime_text_fn=task_runtime_text_fn,
        tail_log_fn=tail_log_fn,
    )


def launch_config(model_name: str, train_dir: str, speech_encoder: str, *, active_task: dict, task_log_dir: Path, task_stage_labels: dict, set_active_task_fn: Callable, task_runtime_text_fn: Callable[[], str], tail_log_fn: Callable[[Path], str]):
    """启动第 2 步：重新生成训练列表和配置文件。"""
    model_name = sanitize_model_name(model_name)
    return start_task(
        "preprocess_flist_config",
        [
            sys.executable,
            "-m", "src.train_pipeline.preprocess_flist_config",
            "--source_dir",
            train_dir,
            "--train_list",
            model_train_list_path(model_name).relative_to(ROOT).as_posix(),
            "--val_list",
            model_val_list_path(model_name).relative_to(ROOT).as_posix(),
            "--config_out",
            model_config_path(model_name).relative_to(ROOT).as_posix(),
            "--diff_config_out",
            model_diff_config_path(model_name).relative_to(ROOT).as_posix(),
            "--exp_dir",
            model_diffusion_dir(model_name).relative_to(ROOT).as_posix(),
            "--speech_encoder",
            speech_encoder,
        ],
        active_task=active_task,
        task_log_dir=task_log_dir,
        task_stage_labels=task_stage_labels,
        set_active_task_fn=set_active_task_fn,
        task_runtime_text_fn=task_runtime_text_fn,
        tail_log_fn=tail_log_fn,
    )


def launch_preprocess(
    model_name: str,
    train_dir: str,
    *,
    default_preprocess_workers: int,
    active_task: dict,
    task_log_dir: Path,
    task_stage_labels: dict,
    set_active_task_fn: Callable,
    task_runtime_text_fn: Callable[[], str],
    tail_log_fn: Callable[[Path], str],
):
    """启动第 3 步：提取内容、音高和频谱等特征。"""
    model_name = sanitize_model_name(model_name)
    return start_task(
        "preprocess_hubert_f0",
        [
            sys.executable,
            "-m", "src.train_pipeline.preprocess_hubert_f0",
            "--in_dir",
            train_dir,
            "--config",
            model_config_path(model_name).relative_to(ROOT).as_posix(),
            "--diff_config",
            model_diff_config_path(model_name).relative_to(ROOT).as_posix(),
            "--num_processes",
            str(default_preprocess_workers),
        ],
        active_task=active_task,
        task_log_dir=task_log_dir,
        task_stage_labels=task_stage_labels,
        set_active_task_fn=set_active_task_fn,
        task_runtime_text_fn=task_runtime_text_fn,
        tail_log_fn=tail_log_fn,
    )


def launch_train(model_name: str, batch_size: int | float | None = None, *, active_task: dict, task_log_dir: Path, task_stage_labels: dict, set_active_task_fn: Callable, task_runtime_text_fn: Callable[[], str], tail_log_fn: Callable[[Path], str]):
    """启动第 4 步：主模型训练。"""
    model_name = sanitize_model_name(model_name)
    ensure_runtime_base_models(model_name)
    _assign_train_port(model_name)
    train_cmd = [
        sys.executable,
        "-m",
        "src.train_pipeline.train",
        "-c",
        model_config_path(model_name).relative_to(ROOT).as_posix(),
        "-m",
        model_name,
    ]
    if batch_size is not None:
        train_cmd.extend(["--batch-size", str(int(batch_size))])
    return start_task(
        "train_main",
        train_cmd,
        active_task=active_task,
        task_log_dir=task_log_dir,
        task_stage_labels=task_stage_labels,
        set_active_task_fn=set_active_task_fn,
        task_runtime_text_fn=task_runtime_text_fn,
        tail_log_fn=tail_log_fn,
    )


def launch_train_diff(model_name: str, *, active_task: dict, task_log_dir: Path, task_stage_labels: dict, set_active_task_fn: Callable, task_runtime_text_fn: Callable[[], str], tail_log_fn: Callable[[Path], str]):
    """启动第 5 步：扩散训练。"""
    model_name = sanitize_model_name(model_name)
    ensure_runtime_base_models(model_name)
    return start_task(
        "train_diff",
        [sys.executable, "-m", "src.train_pipeline.train_diff", "-c", model_diff_config_path(model_name).relative_to(ROOT).as_posix()],
        active_task=active_task,
        task_log_dir=task_log_dir,
        task_stage_labels=task_stage_labels,
        set_active_task_fn=set_active_task_fn,
        task_runtime_text_fn=task_runtime_text_fn,
        tail_log_fn=tail_log_fn,
        display_log_path=model_diffusion_dir(model_name) / "log_info.txt",
    )


def launch_train_index(model_name: str, train_dir: str, *, active_task: dict, task_log_dir: Path, task_stage_labels: dict, set_active_task_fn: Callable, task_runtime_text_fn: Callable[[], str], tail_log_fn: Callable[[Path], str]):
    """启动第 6 步：训练检索增强用的特征索引。"""
    model_name = sanitize_model_name(model_name)
    return start_task(
        "train_index",
        [
            sys.executable,
            "-m", "src.train_pipeline.train_index",
            "-c",
            model_config_path(model_name).relative_to(ROOT).as_posix(),
            "--root_dir",
            train_dir,
            "--output_dir",
            model_root_dir(model_name).relative_to(ROOT).as_posix(),
        ],
        active_task=active_task,
        task_log_dir=task_log_dir,
        task_stage_labels=task_stage_labels,
        set_active_task_fn=set_active_task_fn,
        task_runtime_text_fn=task_runtime_text_fn,
        tail_log_fn=tail_log_fn,
    )


def launch_pipeline_prep(
    model_name: str,
    raw_dir: str,
    train_dir: str,
    speech_encoder: str,
    *,
    default_preprocess_workers: int,
    active_task: dict,
    task_log_dir: Path,
    pipeline_labels: dict,
    task_stage_labels: dict,
    set_active_task_fn: Callable,
    task_runtime_text_fn: Callable[[], str],
    tail_log_fn: Callable[[Path], str],
    append_pipeline_log_fn: Callable[[Path, str], None],
):
    """启动常用准备流程：重采样 -> 配置与列表 -> 提取特征。"""
    model_name = sanitize_model_name(model_name)
    raw_dir = sanitize_dataset_name(raw_dir) or model_name
    train_dir = train_dir or default_train_dir_for_dataset(raw_dir)
    raw_root = ROOT / resolve_raw_dataset_dir(raw_dir)
    _, raw_wavs = count_raw_dataset_wavs(raw_root)
    if raw_wavs == 0:
        return (
            f"无法启动一键执行 1-3 步；当前模型：{model_name}；当前模型数据目录：{resolve_raw_dataset_dir(raw_dir).as_posix()}；状态：未检测到 wav",
            task_runtime_text_fn(),
            tail_log_fn(active_task["log_path"]),
        )
    steps = [
        ("resample", [sys.executable, "-m", "src.train_pipeline.resample", "--in_dir", "training_data/source", "--speaker", raw_dir, "--out_dir2", train_dir], True),
        (
            "preprocess_flist_config",
            [
                sys.executable,
                "-m", "src.train_pipeline.preprocess_flist_config",
                "--source_dir",
                train_dir,
                "--train_list",
                model_train_list_path(model_name).relative_to(ROOT).as_posix(),
                "--val_list",
                model_val_list_path(model_name).relative_to(ROOT).as_posix(),
                "--config_out",
                model_config_path(model_name).relative_to(ROOT).as_posix(),
                "--diff_config_out",
                model_diff_config_path(model_name).relative_to(ROOT).as_posix(),
                "--exp_dir",
                model_diffusion_dir(model_name).relative_to(ROOT).as_posix(),
                "--speech_encoder",
                speech_encoder,
            ],
            True,
        ),
        (
            "preprocess_hubert_f0",
            [
                sys.executable,
                "-m", "src.train_pipeline.preprocess_hubert_f0",
                "--in_dir",
                train_dir,
                "--config",
                model_config_path(model_name).relative_to(ROOT).as_posix(),
                "--diff_config",
                model_diff_config_path(model_name).relative_to(ROOT).as_posix(),
                "--num_processes",
                str(default_preprocess_workers),
            ],
            True,
        ),
    ]
    message, runtime_text, log_text = start_pipeline(
        "pipeline_prep",
        steps,
        active_task=active_task,
        task_log_dir=task_log_dir,
        pipeline_labels=pipeline_labels,
        task_stage_labels=task_stage_labels,
        set_active_task_fn=set_active_task_fn,
        task_runtime_text_fn=task_runtime_text_fn,
        tail_log_fn=tail_log_fn,
        append_pipeline_log_fn=append_pipeline_log_fn,
    )
    return (
        f"{message}\n当前模型：{model_name}；当前模型数据目录：{resolve_raw_dataset_dir(raw_dir).as_posix()}；处理目录：{train_dir}；流程：重采样 -> 配置与列表 -> 提取特征",
        runtime_text,
        log_text,
    )


def launch_pipeline_train_main(
    model_name: str,
    raw_dir: str,
    train_dir: str,
    speech_encoder: str,
    batch_size: int | float | None = None,
    *,
    default_preprocess_workers: int,
    active_task: dict,
    task_log_dir: Path,
    pipeline_labels: dict,
    task_stage_labels: dict,
    set_active_task_fn: Callable,
    task_runtime_text_fn: Callable[[], str],
    tail_log_fn: Callable[[Path], str],
    append_pipeline_log_fn: Callable[[Path, str], None],
):
    """启动到主模型训练为止的完整流程。"""
    model_name = sanitize_model_name(model_name)
    raw_dir = sanitize_dataset_name(raw_dir) or model_name
    train_dir = train_dir or default_train_dir_for_dataset(raw_dir)
    raw_root = ROOT / resolve_raw_dataset_dir(raw_dir)
    _, raw_wavs = count_raw_dataset_wavs(raw_root)
    if raw_wavs == 0:
        return (
            f"无法启动一键执行到主模型训练；当前模型：{model_name}；当前模型数据目录：{resolve_raw_dataset_dir(raw_dir).as_posix()}；状态：未检测到 wav",
            task_runtime_text_fn(),
            tail_log_fn(active_task["log_path"]),
        )
    ensure_runtime_base_models(model_name)
    steps = [
        ("resample", [sys.executable, "-m", "src.train_pipeline.resample", "--in_dir", "training_data/source", "--speaker", raw_dir, "--out_dir2", train_dir], True),
        (
            "preprocess_flist_config",
            [
                sys.executable,
                "-m", "src.train_pipeline.preprocess_flist_config",
                "--source_dir",
                train_dir,
                "--train_list",
                model_train_list_path(model_name).relative_to(ROOT).as_posix(),
                "--val_list",
                model_val_list_path(model_name).relative_to(ROOT).as_posix(),
                "--config_out",
                model_config_path(model_name).relative_to(ROOT).as_posix(),
                "--diff_config_out",
                model_diff_config_path(model_name).relative_to(ROOT).as_posix(),
                "--exp_dir",
                model_diffusion_dir(model_name).relative_to(ROOT).as_posix(),
                "--speech_encoder",
                speech_encoder,
            ],
            True,
        ),
        (
            "preprocess_hubert_f0",
            [
                sys.executable,
                "-m", "src.train_pipeline.preprocess_hubert_f0",
                "--in_dir",
                train_dir,
                "--config",
                model_config_path(model_name).relative_to(ROOT).as_posix(),
                "--diff_config",
                model_diff_config_path(model_name).relative_to(ROOT).as_posix(),
                "--num_processes",
                str(default_preprocess_workers),
            ],
            True,
        ),
        (
            "train_main",
            [
                sys.executable,
                "-m",
                "src.train_pipeline.train",
                "-c",
                model_config_path(model_name).relative_to(ROOT).as_posix(),
                "-m",
                model_name,
            ],
            False,
        ),
    ]
    if batch_size is not None:
        steps[-1][1].extend(["--batch-size", str(int(batch_size))])
    message, runtime_text, log_text = start_pipeline(
        "pipeline_train_main",
        steps,
        active_task=active_task,
        task_log_dir=task_log_dir,
        pipeline_labels=pipeline_labels,
        task_stage_labels=task_stage_labels,
        set_active_task_fn=set_active_task_fn,
        task_runtime_text_fn=task_runtime_text_fn,
        tail_log_fn=tail_log_fn,
        append_pipeline_log_fn=append_pipeline_log_fn,
    )
    return (
        f"{message}\n当前模型：{model_name}；当前模型数据目录：{resolve_raw_dataset_dir(raw_dir).as_posix()}；处理目录：{train_dir}；流程：重采样 -> 配置与列表 -> 提取特征 -> 主模型训练",
        runtime_text,
        log_text,
    )
