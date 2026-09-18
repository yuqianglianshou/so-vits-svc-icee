from src.train_ui.workspace import (
    count_raw_dataset_wavs,
    count_training_wavs,
    speaker_dirs_in_train_root,
)


def test_raw_dataset_counts_only_direct_wav_files(tmp_path):
    """原始数据统计不能把子目录和其他格式误算成训练 WAV。"""
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.WAV").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("note", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.wav").write_bytes(b"")

    assert count_raw_dataset_wavs(tmp_path) == (1, 2)


def test_training_directory_without_direct_wav_is_not_ready(tmp_path):
    """只有子目录而没有直接 WAV 时，不符合当前单说话人训练布局。"""
    nested = tmp_path / "speaker"
    nested.mkdir()
    (nested / "a.wav").write_bytes(b"")

    assert count_training_wavs(tmp_path) == (0, 0)
    assert speaker_dirs_in_train_root(tmp_path) == []


def test_training_directory_reports_its_name_for_direct_wavs(tmp_path):
    """直接包含 WAV 的处理目录应识别为一个可训练说话人。"""
    (tmp_path / "a.wav").write_bytes(b"")

    assert count_training_wavs(tmp_path) == (1, 1)
    assert speaker_dirs_in_train_root(tmp_path) == [tmp_path.name]
