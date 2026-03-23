from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_contains(text: str, needle: str, check_name: str):
    if needle not in text:
        raise AssertionError(f"{check_name}: missing `{needle}`")


def main():
    app_train = ROOT / "app_train.py"
    app_infer = ROOT / "app_infer.py"

    for path in (app_train, app_infer):
        if not path.exists():
            raise AssertionError(f"missing entry file: {path.name}")

    train_text = read_text(app_train)
    infer_text = read_text(app_infer)

    checks = [
        (train_text, "PRETRAIN_ASSETS", "train_pretrain_management"),
        (train_text, "### 训练前依赖", "train_ui_pretrain"),
        (train_text, "from train_ui.paths import", "train_dataset_sanitization"),
        (train_text, "from train_ui.tasks import", "train_task_module_import"),
        (train_text, "delete_dataset_directory", "train_delete_dataset"),
        (train_text, "一键执行 1-3 步", "train_pipeline_prep"),
        (train_text, "一键执行到主模型训练", "train_pipeline_main"),
        (infer_text, "from quality_presets import BEST_QUALITY_PRESET, QUALITY_MODES", "infer_quality_presets"),
        (infer_text, 'label="质量模式"', "infer_quality_mode_ui"),
        (infer_text, "生成运行摘要", "infer_runtime_summary"),
    ]

    for text, needle, check_name in checks:
        assert_contains(text, needle, check_name)

    print("[OK] app_train dependency flow and dataset pipeline flow")
    print("[OK] app_infer quality-mode flow")


if __name__ == "__main__":
    main()
