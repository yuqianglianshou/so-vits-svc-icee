from pathlib import Path

import pytest

from src.infer_ui.files import _resolve_local_model_dir, resolve_uploaded_path


def test_resolve_uploaded_path_accepts_path_and_file_object(tmp_path):
    """Gradio 文件对象和普通路径应归一化为同一种路径值。"""
    path = tmp_path / "model.zip"

    class UploadedFile:
        name = str(path)

    assert resolve_uploaded_path(str(path)) == str(path)
    assert resolve_uploaded_path(UploadedFile()) == str(path)
    assert resolve_uploaded_path(None) == ""


def test_local_model_directory_selects_latest_numeric_checkpoint(tmp_path):
    """未手选 checkpoint 时应选择步数最大的生成器文件。"""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "G_900.pth").write_bytes(b"")
    (tmp_path / "G_1000.pth").write_bytes(b"")

    checkpoint, config = _resolve_local_model_dir(str(tmp_path))

    assert checkpoint == str(tmp_path / "G_1000.pth")
    assert config == str(tmp_path / "config.json")


def test_local_model_directory_requires_config_and_checkpoint(tmp_path):
    """不完整的模型目录必须给出明确错误，不能静默选择其他文件。"""
    with pytest.raises(FileNotFoundError, match="缺少 config.json"):
        _resolve_local_model_dir(str(tmp_path))

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"缺少 G_\*\.pth"):
        _resolve_local_model_dir(str(tmp_path))


def test_selected_checkpoint_must_belong_to_model_directory(tmp_path):
    """手选其他目录的 checkpoint 时必须拒绝加载。"""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "G_100.pth").write_bytes(b"")
    foreign = Path(tmp_path.parent) / "G_200.pth"

    with pytest.raises(FileNotFoundError, match="不属于当前本地模型目录"):
        _resolve_local_model_dir(str(tmp_path), str(foreign))
