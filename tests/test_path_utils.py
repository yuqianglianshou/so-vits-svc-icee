from src import path_utils


def test_dependency_paths_remain_under_dependency_root():
    """依赖路径不能意外写入模型工作区或项目根目录。"""
    dependency_root = path_utils.ROOT / "model_assets" / "dependencies"
    paths = (
        path_utils.get_contentvec_hf_path(),
        path_utils.get_rmvpe_path(),
        path_utils.get_nsf_hifigan_model_path(),
        path_utils.get_sovits_g0_path(),
        path_utils.get_sovits_d0_path(),
        path_utils.get_diffusion_model_0_path(),
    )

    assert all(dependency_root in path.parents for path in paths)


def test_ensure_runtime_base_models_preserves_expected_names(tmp_path, monkeypatch):
    """底模物化到工作区时必须保持训练代码约定的文件名。"""
    dependency_root = tmp_path / "dependencies" / "44k"
    monkeypatch.setattr(path_utils, "ROOT", tmp_path)
    monkeypatch.setattr(path_utils, "BASE_MODEL_44K_DIR", dependency_root)
    monkeypatch.setattr(
        path_utils,
        "BASE_MODEL_44K_DIFFUSION_DIR",
        dependency_root / "diffusion",
    )
    for source in (
        path_utils.get_sovits_g0_path(),
        path_utils.get_sovits_d0_path(),
        path_utils.get_diffusion_model_0_path(),
    ):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"model")

    path_utils.ensure_runtime_base_models("demo")

    workspace = tmp_path / "model_assets" / "workspaces" / "demo"
    assert (workspace / "G_0.pth").read_bytes() == b"model"
    assert (workspace / "D_0.pth").read_bytes() == b"model"
    assert (workspace / "diffusion" / "model_0.pt").read_bytes() == b"model"
