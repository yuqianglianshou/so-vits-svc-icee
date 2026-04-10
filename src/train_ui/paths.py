"""训练页共用的路径与命名规则。

这个模块只负责两类事情：
1. 规范训练页里出现的模型名、数据目录名，
2. 统一生成训练相关文件和目录路径。

这样状态判断模块和页面入口都可以复用同一套规则，避免各自维护。
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sanitize_dataset_name(dataset_name: str):
    """清洗训练语音数据目录名，避免把非法路径写入页面状态。"""
    name = (dataset_name or "").strip()
    if not name:
        return None
    if name in {".", ".."}:
        return None
    invalid_parts = ["..", "/", "\\", ":"]
    if any(part in name for part in invalid_parts):
        return None
    if Path(name).name != name:
        return None
    return name


def resolve_raw_dataset_dir(dataset_name: str):
    """把训练语音数据名映射到 training_data/source 下的真实目录。"""
    name = sanitize_dataset_name(dataset_name)
    if not name:
        name = "default_dataset"
    return Path("training_data/source") / name


def default_train_dir_for_dataset(dataset_name: str):
    """返回当前训练语音数据对应的默认重采样输出目录。"""
    name = sanitize_dataset_name(dataset_name)
    if not name:
        name = "default_dataset"
    return f"training_data/processed/44k/{name}"


def sanitize_model_name(model_name: str):
    """清洗模型工作区名称，避免把非法路径带入 model_assets/workspaces/ 结构。"""
    name = (model_name or "").strip()
    if not name:
        return "default_model"
    if name in {".", ".."}:
        return "default_model"
    invalid_parts = ["..", "/", "\\", ":"]
    if any(part in name for part in invalid_parts):
        return "default_model"
    if Path(name).name != name:
        return "default_model"
    return name


def model_root_dir(model_name: str) -> Path:
    """返回某个训练模型在 model_assets/workspaces/ 下的工作区目录。"""
    return ROOT / "model_assets/workspaces" / sanitize_model_name(model_name)


def model_diffusion_dir(model_name: str) -> Path:
    """返回扩散训练产物目录。"""
    return model_root_dir(model_name) / "diffusion"


def model_config_path(model_name: str) -> Path:
    """返回主模型训练配置文件路径。"""
    return model_root_dir(model_name) / "config.json"


def model_diff_config_path(model_name: str) -> Path:
    """返回扩散训练配置文件路径。"""
    return model_root_dir(model_name) / "diffusion.yaml"


def model_train_list_path(model_name: str) -> Path:
    """返回训练集 filelist 路径。"""
    return model_root_dir(model_name) / "filelists" / "train.txt"


def model_val_list_path(model_name: str) -> Path:
    """返回验证集 filelist 路径。"""
    return model_root_dir(model_name) / "filelists" / "val.txt"


def model_index_path(model_name: str) -> Path:
    """返回音色增强索引输出路径。"""
    return model_root_dir(model_name) / "feature_and_index.pkl"


def model_workspace_path(model_name: str) -> Path:
    """返回模型工作区记录文件路径。"""
    return model_root_dir(model_name) / "workspace.json"
