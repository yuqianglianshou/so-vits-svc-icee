from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
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


def launch_tensorboard(*, root: Path, active_task: dict, ui_notice: dict) -> str:
    tb_cmd = [sys.executable, "-m", "tensorboard.main", "--logdir=model_assets/workspaces", "--port=6006"]
    current = active_task["proc"]
    if current is not None and current.poll() is None:
        return "当前已有训练任务在运行。TensorBoard 不占用训练槽位，请手动在终端启动。"
    subprocess.Popen(tb_cmd, cwd=root, start_new_session=True)
    opened = open_local_url("http://127.0.0.1:6006", root=root)
    if opened:
        message = "已启动并尝试打开训练监控：http://127.0.0.1:6006"
    else:
        message = "训练监控已启动，但浏览器没有成功打开。请手动访问：http://127.0.0.1:6006"
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
        set_ui_notice(ui_notice, message)
        return message
    except Exception as exc:
        message = f"打开推理页面失败：{type(exc).__name__}: {exc}"
        set_ui_notice(ui_notice, message)
        return message
