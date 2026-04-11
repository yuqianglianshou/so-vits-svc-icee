from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr
import soundfile as sf


def list_input_audio_history(history_dir: Path):
    history_dir.mkdir(parents=True, exist_ok=True)
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    files = [
        path.as_posix()
        for path in history_dir.iterdir()
        if path.is_file() and path.suffix.lower() in audio_exts and not path.name.startswith(".")
    ]
    files.sort(key=lambda item: Path(item).stat().st_mtime, reverse=True)
    return files


def build_input_audio_history_choices(history_dir: Path):
    return [(Path(path).name, path) for path in list_input_audio_history(history_dir)]


def refresh_input_audio_history(history_dir: Path):
    choices = build_input_audio_history_choices(history_dir)
    return gr.update(choices=choices, interactive=bool(choices))


def save_input_audio_to_history(history_dir: Path, audio_path):
    if not audio_path:
        return None
    source = Path(audio_path)
    if not source.exists():
        return None
    history_dir.mkdir(parents=True, exist_ok=True)
    target = history_dir / source.name
    stem = source.stem
    suffix = source.suffix
    index = 2
    while target.exists() and target.resolve() != source.resolve():
        target = history_dir / f"{stem}_{index}{suffix}"
        index += 1
    if target.resolve() != source.resolve():
        shutil.copy2(source, target)
    return target.as_posix()


def describe_input_audio(audio_path):
    if not audio_path:
        return '<div class="subtle-note">尚未选择输入音频。</div>'
    try:
        info = sf.info(audio_path)
        minutes = int(info.duration // 60)
        seconds = int(round(info.duration % 60))
        if seconds == 60:
            minutes += 1
            seconds = 0
        duration_text = f"{minutes}:{seconds:02d}"
        return (
            '<div class="subtle-note">'
            f'后端识别时长：{duration_text}（{info.duration:.2f} 秒）；'
            f'采样率：{info.samplerate} Hz；'
            f'声道：{info.channels}；'
            f'格式：{info.format}'
            '</div>'
        )
    except Exception as exc:
        return f'<div class="subtle-note">无法读取输入音频信息：{type(exc).__name__}: {exc}</div>'


def preview_input_audio(audio_path):
    return audio_path or None


def handle_uploaded_input_audio(history_dir: Path, audio_path):
    saved_path = save_input_audio_to_history(history_dir, audio_path)
    choices = list_input_audio_history(history_dir)
    info_html = describe_input_audio(saved_path)
    preview_value = preview_input_audio(saved_path)
    return (
        gr.update(choices=choices, value=saved_path, interactive=bool(choices)),
        info_html,
        preview_value,
    )


def handle_selected_history_audio(audio_path):
    resolved_path = audio_path or None
    return describe_input_audio(resolved_path), preview_input_audio(resolved_path)
