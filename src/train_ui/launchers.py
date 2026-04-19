from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def open_local_url(url: str, *, root: Path) -> bool:
    try:
        if webbrowser.open(url):
            return True
    except Exception:
        pass

    system_name = platform.system()
    try:
        if system_name == "Darwin":
            subprocess.Popen(["/usr/bin/open", url], cwd=root, start_new_session=True)
            return True
        if system_name == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        if system_name == "Linux":
            subprocess.Popen(["/usr/bin/xdg-open", url], cwd=root, start_new_session=True)
            return True
    except Exception:
        return False
    return False


def set_ui_notice(ui_notice: dict, message: str, ttl_seconds: int = 15):
    ui_notice["message"] = message
    ui_notice["expires_at"] = time.time() + ttl_seconds


def find_available_port(default_port: int, max_tries: int = 100) -> int:
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        fallback_port = sock.getsockname()[1]
        if fallback_port:
            return fallback_port
    raise OSError(f"无法在 {default_port}-{default_port + max_tries - 1} 范围内找到可用端口。")


def wait_for_local_url_and_open(
    *,
    proc: subprocess.Popen,
    port: int,
    url: str,
    root: Path,
    state: dict,
    ui_notice: dict,
    page_label: str,
    timeout_seconds: int = 90,
):
    """在页面启动较慢时后台等待端口就绪，再尝试打开浏览器。"""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            set_ui_notice(ui_notice, f"{page_label}启动失败：进程已退出（退出码 {proc.returncode}）。", ttl_seconds=60)
            return
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                state["proc"] = proc
                state["port"] = port
                if open_local_url(url, root=root):
                    set_ui_notice(ui_notice, f"已启动并打开{page_label}：{url}", ttl_seconds=30)
                else:
                    set_ui_notice(ui_notice, f"{page_label}已启动，但浏览器没有成功打开。请手动访问：{url}", ttl_seconds=60)
                return
        time.sleep(0.5)
    set_ui_notice(ui_notice, f"{page_label}仍在启动中：{url}；如果浏览器没有自动打开，可手动访问这个地址。", ttl_seconds=60)


def ensure_localhost_bypass_proxy():
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


def launch_tensorboard(*, root: Path, tensorboard_state: dict, ui_notice: dict) -> str:
    try:
        tensorboard_port = 6006
        tensorboard_url = f"http://127.0.0.1:{tensorboard_port}"
        existing_proc = tensorboard_state.get("proc")
        existing_port = tensorboard_state.get("port") or tensorboard_port

        if existing_proc is not None and existing_proc.poll() is None:
            existing_url = f"http://127.0.0.1:{existing_port}"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", existing_port)) == 0:
                    if open_local_url(existing_url, root=root):
                        message = f"已打开当前训练监控：{existing_url}"
                    else:
                        message = f"训练监控已在运行，请手动访问：{existing_url}"
                    set_ui_notice(ui_notice, message)
                    return message

        tb_cmd = [sys.executable, "-m", "tensorboard.main", "--logdir=model_assets/workspaces", "--port", str(tensorboard_port)]
        proc = subprocess.Popen(
            tb_cmd,
            cwd=root,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tensorboard_state["proc"] = proc
        tensorboard_state["port"] = tensorboard_port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", tensorboard_port)) == 0:
                if open_local_url(tensorboard_url, root=root):
                    message = f"已启动并打开训练监控：{tensorboard_url}"
                else:
                    message = f"训练监控已启动，但浏览器没有成功打开。请手动访问：{tensorboard_url}"
                set_ui_notice(ui_notice, message)
                return message

        message = f"训练监控正在启动中：{tensorboard_url}；如果浏览器没有自动打开，可手动访问这个地址。"
        set_ui_notice(ui_notice, message, ttl_seconds=60)
        threading.Thread(
            target=wait_for_local_url_and_open,
            kwargs={
                "proc": proc,
                "port": tensorboard_port,
                "url": tensorboard_url,
                "root": root,
                "state": tensorboard_state,
                "ui_notice": ui_notice,
                "page_label": "训练监控",
            },
            daemon=True,
        ).start()
        return message
    except Exception as exc:
        message = f"打开训练监控失败：{type(exc).__name__}: {exc}"
        set_ui_notice(ui_notice, message)
        return message


def launch_infer_ui(*, root: Path, infer_ui_state: dict, ui_notice: dict) -> str:
    try:
        existing_proc = infer_ui_state["proc"]
        existing_port = infer_ui_state["port"]
        if existing_proc is not None and existing_proc.poll() is None and existing_port:
            infer_url = f"http://127.0.0.1:{existing_port}"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", existing_port)) == 0:
                    if open_local_url(infer_url, root=root):
                        message = f"已打开当前会话的推理页面：{infer_url}"
                    else:
                        message = f"推理页面已就绪，但浏览器没有成功打开。请手动访问：{infer_url}"
                    set_ui_notice(ui_notice, message)
                    return message

        env = os.environ.copy()
        env["OPEN_BROWSER"] = "0"
        infer_port = find_available_port(7860)
        infer_url = f"http://127.0.0.1:{infer_port}"
        env["GRADIO_SERVER_PORT"] = str(infer_port)
        env["GRADIO_ANALYTICS_ENABLED"] = "False"
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.app_infer"],
            cwd=root,
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(20):
            if proc.poll() is not None:
                return f"推理页面启动失败：src.app_infer 已退出（退出码 {proc.returncode}）。请先在终端单独运行并查看报错。"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", infer_port)) == 0:
                    infer_ui_state["proc"] = proc
                    infer_ui_state["port"] = infer_port
                    if open_local_url(infer_url, root=root):
                        message = f"已启动并打开推理页面：{infer_url}"
                    else:
                        message = f"推理页面已启动，但浏览器没有成功打开。请手动访问：{infer_url}"
                    set_ui_notice(ui_notice, message)
                    return message
            time.sleep(0.25)

        infer_ui_state["proc"] = proc
        infer_ui_state["port"] = infer_port
        message = f"推理页面正在启动中：{infer_url}；如果浏览器没有自动打开，可手动访问这个地址。"
        set_ui_notice(ui_notice, message, ttl_seconds=60)
        threading.Thread(
            target=wait_for_local_url_and_open,
            kwargs={
                "proc": proc,
                "port": infer_port,
                "url": infer_url,
                "root": root,
                "state": infer_ui_state,
                "ui_notice": ui_notice,
                "page_label": "推理页面",
            },
            daemon=True,
        ).start()
        return message
    except Exception as exc:
        message = f"打开推理页面失败：{type(exc).__name__}: {exc}"
        set_ui_notice(ui_notice, message)
        return message
