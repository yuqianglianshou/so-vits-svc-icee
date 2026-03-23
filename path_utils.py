from pathlib import Path
import os
import shutil


ROOT = Path(__file__).resolve().parent

PRETRAIN_DIR = ROOT / "pretrain"
ENCODER_DIR = PRETRAIN_DIR / "encoders"
VOCODER_DIR = PRETRAIN_DIR / "vocoders"
BASE_MODEL_DIR = PRETRAIN_DIR / "base_models"
BASE_MODEL_44K_DIR = BASE_MODEL_DIR / "44k"
BASE_MODEL_44K_DIFFUSION_DIR = BASE_MODEL_44K_DIR / "diffusion"


def get_contentvec_path() -> Path:
    """兼容旧调用名，当前统一返回 HF ContentVec 目录。"""
    return get_contentvec_hf_path()


def get_contentvec_hf_path() -> Path:
    return ENCODER_DIR / "contentvec_hf"


def get_rmvpe_path() -> Path:
    return ENCODER_DIR / "rmvpe.pt"


def get_nsf_hifigan_dir() -> Path:
    return VOCODER_DIR / "nsf_hifigan"


def get_nsf_hifigan_model_path() -> Path:
    return get_nsf_hifigan_dir() / "model"


def get_nsf_hifigan_config_path() -> Path:
    return get_nsf_hifigan_dir() / "config.json"


def get_sovits_g0_path() -> Path:
    return BASE_MODEL_44K_DIR / "G_0.pth"


def get_sovits_d0_path() -> Path:
    return BASE_MODEL_44K_DIR / "D_0.pth"


def get_diffusion_model_0_path() -> Path:
    return BASE_MODEL_44K_DIFFUSION_DIR / "model_0.pt"


def _materialize_reference(source: Path, target: Path):
    if target.exists() or not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return True
    except OSError:
        pass
    try:
        target.symlink_to(source)
        return True
    except OSError:
        pass
    shutil.copy2(source, target)
    return True


def ensure_runtime_base_models(model_name: str = "44k"):
    runtime_dir = ROOT / "logs" / model_name
    runtime_diff_dir = runtime_dir / "diffusion"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_diff_dir.mkdir(parents=True, exist_ok=True)

    copies = []
    mappings = [
        (get_sovits_g0_path(), runtime_dir / "G_0.pth"),
        (get_sovits_d0_path(), runtime_dir / "D_0.pth"),
        (get_diffusion_model_0_path(), runtime_diff_dir / "model_0.pt"),
    ]
    for source, target in mappings:
        if _materialize_reference(source, target):
            copies.append((source, target))
    return copies
