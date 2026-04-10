#!/usr/bin/env python3
"""Lightweight regression checks for recent stability fixes."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def assert_contains(text: str, needle: str, msg: str) -> None:
    if needle not in text:
        raise AssertionError(msg)


def assert_not_contains(text: str, needle: str, msg: str) -> None:
    if needle in text:
        raise AssertionError(msg)


def check_infer_unpacking() -> None:
    infer_tool = read_text("inference/infer_tool.py")
    flask_api = read_text("services/flask_api.py")
    flask_api_full = read_text("services/flask_api_full_song.py")

    assert_contains(
        infer_tool,
        "audio, _, _ = svc_model.infer(",
        "RealTimeVC should unpack three return values from svc_model.infer().",
    )
    assert_contains(
        flask_api,
        "out_audio, _, _ = svc_model.infer(",
        "flask_api should unpack three return values from svc_model.infer().",
    )
    assert_contains(
        flask_api_full,
        "out_audio, _, _ = svc_model.infer(",
        "flask_api_full_song should unpack three return values from svc_model.infer().",
    )


def check_speaker_resolution() -> None:
    infer_tool = read_text("inference/infer_tool.py")

    assert_contains(
        infer_tool,
        "def _resolve_speaker_id(self, speaker):",
        "Svc should have a dedicated speaker-id resolver.",
    )
    assert_contains(
        infer_tool,
        "speaker_id = self._resolve_speaker_id(speaker)",
        "Svc should use _resolve_speaker_id in infer paths.",
    )
    assert_not_contains(
        infer_tool,
        "self.spk2id.__dict__",
        "Speaker resolution should not use dict.__dict__.",
    )


def check_ddp_sampler() -> None:
    train = read_text("train_pipeline/train.py")

    assert_contains(
        train,
        "from torch.utils.data.distributed import DistributedSampler",
        "train_pipeline/train.py should import DistributedSampler.",
    )
    assert_contains(
        train,
        "train_sampler = DistributedSampler(",
        "train_pipeline/train.py should create DistributedSampler for training dataset.",
    )
    assert_contains(
        train,
        "train_sampler.set_epoch(epoch)",
        "train_pipeline/train.py should reseed sampler every epoch.",
    )


def check_preprocess_parallel() -> None:
    preprocess = read_text("train_pipeline/preprocess_hubert_f0.py")

    assert_contains(
        preprocess,
        "def process_batch(",
        "process_batch signature should carry config paths instead of mel_extractor object.",
    )
    assert_contains(
        preprocess,
        "mel_extractor = Vocoder(",
        "process_batch should initialize mel_extractor inside worker.",
    )
    assert_contains(
        preprocess,
        "def parallel_process(filenames, num_processes, f0p, diff, device, config_path, diff_config_path):",
        "parallel_process signature should pass config paths instead of mel_extractor object.",
    )
    assert_contains(
        preprocess,
        "executor.submit(",
        "parallel_process should still submit batch tasks through ProcessPoolExecutor.",
    )
    assert_contains(
        preprocess,
        "config_path=config_path",
        "parallel_process should forward main config path into worker.",
    )
    assert_contains(
        preprocess,
        "diff_config_path=diff_config_path",
        "parallel_process should forward diffusion config path into worker.",
    )


def main() -> int:
    checks = [
        ("infer_unpacking", check_infer_unpacking),
        ("speaker_resolution", check_speaker_resolution),
        ("ddp_sampler", check_ddp_sampler),
        ("preprocess_parallel", check_preprocess_parallel),
    ]
    for name, check in checks:
        check()
        print(f"[OK] {name}")
    print("All stability-fix checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
