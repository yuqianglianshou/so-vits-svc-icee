from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

from src.infer_ui.local_models import IMPORTED_MODEL_ROOT
from src.train_ui.paths import sanitize_model_name


BUNDLE_TYPE = "so_vits_infer_bundle"
BUNDLE_VERSION = 1


def is_infer_bundle_path(path_like) -> bool:
    path = Path(str(path_like or "")).expanduser()
    return path.is_file() and path.suffix == ".pth"


def _read_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _infer_model_name(config_payload: dict, fallback_name: str) -> str:
    speakers = list((config_payload.get("spk") or {}).keys())
    if len(speakers) == 1:
        candidate = sanitize_model_name(speakers[0])
        if candidate and candidate != "default_model":
            return candidate
    fallback = sanitize_model_name(fallback_name)
    return fallback or "uploaded_model"


def _diffusion_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (TypeError, ValueError):
        return -1


def _list_trained_diffusion_checkpoints(diffusion_dir: Path):
    return [
        path
        for path in sorted(diffusion_dir.glob("model_*.pt"), key=_diffusion_step)
        if path.name != "model_0.pt"
    ]


def _find_diffusion_artifacts(model_dir: Path) -> tuple[Path | None, Path | None]:
    diff_config_path = model_dir / "diffusion.yaml"
    diffusion_dir = model_dir / "diffusion"
    diff_model_path = None
    if diffusion_dir.is_dir():
        candidates = _list_trained_diffusion_checkpoints(diffusion_dir)
        if candidates:
            diff_model_path = candidates[-1]
    if not diff_config_path.is_file() or diff_model_path is None:
        return None, None
    return diff_model_path, diff_config_path


def _find_index_artifact(model_dir: Path) -> Path | None:
    index_path = model_dir / "feature_and_index.pkl"
    return index_path if index_path.is_file() else None


def _find_cover_artifact(model_dir: Path) -> Path | None:
    candidates = [
        model_dir / "cover.png",
        model_dir / "cover.jpg",
        model_dir / "cover.jpeg",
        model_dir / "cover.webp",
        model_dir / f"{model_dir.name}.png",
        model_dir / f"{model_dir.name}.jpg",
        model_dir / f"{model_dir.name}.jpeg",
        model_dir / f"{model_dir.name}.webp",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _find_description_artifact(model_dir: Path) -> Path | None:
    candidates = [
        model_dir / "description.md",
        model_dir / "description.txt",
        model_dir / "README.md",
        model_dir / "README.txt",
        model_dir / "说明.md",
        model_dir / "说明.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _remove_if_exists(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _cleanup_managed_bundle_artifacts(unpack_dir: Path):
    _remove_if_exists(unpack_dir / "diffusion")
    _remove_if_exists(unpack_dir / "diffusion.yaml")
    _remove_if_exists(unpack_dir / "feature_and_index.pkl")
    for candidate in (
        unpack_dir / "cover.png",
        unpack_dir / "cover.jpg",
        unpack_dir / "cover.jpeg",
        unpack_dir / "cover.webp",
        unpack_dir / "description.md",
        unpack_dir / "description.txt",
        unpack_dir / "README.md",
        unpack_dir / "README.txt",
        unpack_dir / "说明.md",
        unpack_dir / "说明.txt",
    ):
        _remove_if_exists(candidate)


def build_infer_bundle_payload(model_path: Path, config_path: Path) -> dict:
    config_payload = _read_config(config_path)
    model_state = torch.load(model_path, map_location="cpu")
    model_name = _infer_model_name(config_payload, model_path.stem)
    model_dir = config_path.parent
    diffusion_model_path, diffusion_config_path = _find_diffusion_artifacts(config_path.parent)
    index_path = _find_index_artifact(model_dir)
    cover_path = _find_cover_artifact(model_dir)
    description_path = _find_description_artifact(model_dir)
    diffusion_payload = None
    if diffusion_model_path and diffusion_config_path:
        diffusion_payload = {
            "filename": diffusion_model_path.name,
            "checkpoint": torch.load(diffusion_model_path, map_location="cpu"),
            "config_filename": diffusion_config_path.name,
            "config_text": _read_text(diffusion_config_path),
        }
    index_payload = None
    if index_path:
        index_payload = {
            "filename": index_path.name,
            "bytes": index_path.read_bytes(),
        }
    cover_payload = None
    if cover_path:
        cover_payload = {
            "filename": cover_path.name,
            "bytes": cover_path.read_bytes(),
        }
    description_payload = None
    if description_path:
        description_payload = {
            "filename": description_path.name,
            "text": _read_text(description_path),
        }
    return {
        "bundle_type": BUNDLE_TYPE,
        "bundle_version": BUNDLE_VERSION,
        "meta": {
            "model_name": model_name,
            "generator_filename": model_path.name,
            "created_at": int(time.time()),
            "sample_rate": config_payload.get("data", {}).get("sampling_rate"),
            "speech_encoder": config_payload.get("model", {}).get("speech_encoder"),
            "speaker_names": list((config_payload.get("spk") or {}).keys()),
            "has_diffusion": diffusion_payload is not None,
            "has_index": index_payload is not None,
            "has_cover": cover_payload is not None,
            "has_description": description_payload is not None,
        },
        "config": config_payload,
        "generator": {
            "filename": model_path.name,
            "checkpoint": model_state,
        },
        "diffusion": diffusion_payload,
        "index": index_payload,
        "cover": cover_payload,
        "description": description_payload,
    }


def export_infer_bundle(model_path: Path, config_path: Path, output_path: Path) -> tuple[Path, dict]:
    payload = build_infer_bundle_payload(model_path, config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    meta = payload.get("meta") or {}
    return output_path, {
        "has_diffusion": bool(meta.get("has_diffusion")),
        "has_index": bool(meta.get("has_index")),
        "has_cover": bool(meta.get("has_cover")),
        "has_description": bool(meta.get("has_description")),
    }


def export_local_model_bundle(local_model_selection: str, checkpoint_selection: str) -> tuple[Path, dict]:
    model_dir = Path((local_model_selection or "").strip()).expanduser()
    checkpoint_path = Path((checkpoint_selection or "").strip()).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在：{model_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"主模型 G_*.pth 不存在：{checkpoint_path}")
    if checkpoint_path.parent != model_dir:
        raise ValueError("所选 G_*.pth 不属于当前本地模型目录。")

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"本地模型目录缺少 config.json：{model_dir}")

    output_path = model_dir / f"{model_dir.name}_{checkpoint_path.stem}_infer_bundle.pth"
    return export_infer_bundle(checkpoint_path, config_path, output_path)


def inspect_infer_bundle_extras(bundle_path_like) -> dict:
    bundle_path = Path(str(bundle_path_like or "")).expanduser()
    if not bundle_path.is_file():
        return {
            "has_diffusion": False,
            "has_index": False,
            "has_cover": False,
            "has_description": False,
        }

    try:
        payload = torch.load(bundle_path, map_location="cpu")
    except Exception:
        return {
            "has_diffusion": False,
            "has_index": False,
            "has_cover": False,
            "has_description": False,
        }
    if payload.get("bundle_type") != BUNDLE_TYPE:
        return {
            "has_diffusion": False,
            "has_index": False,
            "has_cover": False,
            "has_description": False,
        }
    meta = payload.get("meta") or {}
    return {
        "has_diffusion": bool(meta.get("has_diffusion")),
        "has_index": bool(meta.get("has_index")),
        "has_cover": bool(meta.get("has_cover")),
        "has_description": bool(meta.get("has_description")),
    }


def _bundle_hash(bundle_path: Path) -> str:
    digest = hashlib.sha256()
    with bundle_path.open("rb") as bundle_file:
        for chunk in iter(lambda: bundle_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:10]


def unpack_infer_bundle(bundle_path_like) -> tuple[str, str, str, str, str, str]:
    bundle_path = Path(str(bundle_path_like or "")).expanduser()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"推理包不存在：{bundle_path}")

    payload = torch.load(bundle_path, map_location="cpu")
    if payload.get("bundle_type") != BUNDLE_TYPE:
        raise ValueError("上传文件不是受支持的单文件推理包。")
    if int(payload.get("bundle_version", 0)) != BUNDLE_VERSION:
        raise ValueError(f"暂不支持推理包版本：{payload.get('bundle_version')}")

    config_payload = payload.get("config")
    generator_payload = payload.get("generator") or {}
    diffusion_payload = payload.get("diffusion") or {}
    index_payload = payload.get("index") or {}
    cover_payload = payload.get("cover") or {}
    description_payload = payload.get("description") or {}
    generator_filename = str(generator_payload.get("filename") or "").strip() or "G_bundle.pth"
    checkpoint_payload = generator_payload.get("checkpoint")
    if not isinstance(config_payload, dict):
        raise ValueError("推理包缺少有效的 config 配置。")
    if checkpoint_payload is None:
        raise ValueError("推理包缺少主模型权重。")

    model_name = _infer_model_name(config_payload, bundle_path.stem.replace("_infer_bundle", ""))
    unpack_dir = Path(IMPORTED_MODEL_ROOT) / model_name
    unpack_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_managed_bundle_artifacts(unpack_dir)

    bundle_copy_path = unpack_dir / bundle_path.name
    if bundle_copy_path.resolve() != bundle_path.resolve():
        bundle_copy_path.write_bytes(bundle_path.read_bytes())

    config_out = unpack_dir / "config.json"
    generator_out = unpack_dir / generator_filename
    config_out.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(checkpoint_payload, generator_out)

    diffusion_model_out = ""
    diffusion_config_out = ""
    index_out = ""
    if diffusion_payload:
        diff_checkpoint = diffusion_payload.get("checkpoint")
        diff_filename = str(diffusion_payload.get("filename") or "").strip() or "model_0.pt"
        diff_config_text = diffusion_payload.get("config_text")
        diff_config_filename = str(diffusion_payload.get("config_filename") or "").strip() or "diffusion.yaml"
        if diff_checkpoint is not None and diff_config_text:
            diffusion_dir = unpack_dir / "diffusion"
            diffusion_dir.mkdir(parents=True, exist_ok=True)
            diffusion_model_path = diffusion_dir / diff_filename
            diffusion_config_path = unpack_dir / diff_config_filename
            torch.save(diff_checkpoint, diffusion_model_path)
            diffusion_config_path.write_text(str(diff_config_text), encoding="utf-8")
            diffusion_model_out = str(diffusion_model_path)
            diffusion_config_out = str(diffusion_config_path)

    if index_payload:
        index_bytes = index_payload.get("bytes")
        index_filename = str(index_payload.get("filename") or "").strip() or "feature_and_index.pkl"
        if index_bytes:
            index_path = unpack_dir / index_filename
            index_path.write_bytes(index_bytes)
            index_out = str(index_path)

    if cover_payload:
        cover_bytes = cover_payload.get("bytes")
        cover_filename = str(cover_payload.get("filename") or "").strip() or "cover.png"
        if cover_bytes:
            (unpack_dir / cover_filename).write_bytes(cover_bytes)

    if description_payload:
        description_text = description_payload.get("text")
        description_filename = str(description_payload.get("filename") or "").strip() or "description.md"
        if description_text:
            (unpack_dir / description_filename).write_text(str(description_text), encoding="utf-8")

    meta_path = unpack_dir / "bundle_meta.json"
    meta = {
        "source_bundle": bundle_path.name,
        "bundle_hash": _bundle_hash(bundle_path),
        "updated_at": int(time.time()),
        "has_diffusion": bool(diffusion_model_out and diffusion_config_out),
        "has_index": bool(index_out),
        "has_cover": bool(cover_payload),
        "has_description": bool(description_payload),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(generator_out), str(config_out), str(unpack_dir), diffusion_model_out, diffusion_config_out, index_out
