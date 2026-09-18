# So-VITS-SVC Icee

English | [简体中文](./README.md)

[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](./LICENSE)

A local, offline visual workspace for training So-VITS-SVC models and converting singing voices. It is based on so-vits-svc 4.1 and focuses on a clearer single-speaker workflow, dependency preparation, model management, and inference.

> This repository is currently a release candidate. The model core and file formats remain on the so-vits-svc 4.1 compatibility path. Windows with an NVIDIA GPU is the primary supported environment.

## Responsible Use

Use this project only for learning, research, and legally authorized scenarios. You are responsible for the rights to all training data, input audio, models, and generated audio. Do not use it for illegal activity, infringement, impersonation, harassment, fraud, or political or religious manipulation.

This repository does not provide trained character or real-person models and is not responsible for models or content created by users.

## Differences from Upstream 4.1

| Area | so-vits-svc 4.1 | This project |
| --- | --- | --- |
| Training | CLI and legacy WebUI | Workspace-based visual training page |
| Inference | Files and options are spread across tools | Model import, quality presets, and output management |
| Dependencies | Mostly placed manually | Status checks, guided downloads, and manual import |
| Data layout | Shared global directories | An isolated workspace for each model |
| Primary audience | Users familiar with the repository internals | Local Windows users |
| Compatibility | so-vits-svc 4.1 | Keeps the 4.1 model compatibility path |

The project does not claim a new SVC architecture. Its current value is a more understandable workflow, organized storage, dependency management, and safer defaults.

## Screenshots

Training page:

![Training page](./images/训练页.png)

Inference page:

![Inference page](./images/推理页.png)

## Supported Scope

Primary target:

- Windows 10/11 64-bit.
- NVIDIA GPU with CUDA.
- Python 3.11.
- Single-speaker, single-model workspaces.
- Local offline singing voice conversion.
- The 44.1 kHz path using ContentVec `vec768l12` and RMVPE.

Linux has not completed the same release validation. macOS is not recommended for production training. Multi-speaker training, real-time conversion, and ONNX are not primary supported workflows in this release.

## Quick Install

Install Git, Python 3.11, FFmpeg, and a compatible NVIDIA driver. From the repository root:

```powershell
python -m venv .venv311
.venv311\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch and torchaudio for your CUDA environment, then install the repository dependencies:

```powershell
pip install -r requirements.txt
```

Use the [official PyTorch selector](https://pytorch.org/get-started/locally/) for the appropriate installation command.

## Launch

Double-click the scripts under `launchers/`, or run:

```powershell
python -m src.app_train
python -m src.app_infer
```

## Shortest Training Path

1. Launch the training page.
2. Complete the required dependency and base-model checks.
3. Create a model workspace.
4. Import clean and legally authorized WAV files.
5. Run resampling, configuration generation, and feature extraction.
6. Start main-model training.
7. Optionally train shallow diffusion and a feature-retrieval index.
8. Load the resulting model on the inference page and run a conversion.

## Required Assets

The training page checks and guides the installation of ContentVec HF, RMVPE, `G_0.pth`, `D_0.pth`, the optional diffusion base model, and NSF-HiFiGAN. Prefer the guided download flow on the training page.

Detailed user documentation is currently maintained primarily in Simplified Chinese:

- [Documentation index](./docs/README.md)
- [Quick start](./docs/00_快速开始.md)
- [Environment and compatibility](./docs/07_环境与兼容性.md)
- [Model asset inventory](./docs/08_模型依赖清单.md)
- [Known limitations](./docs/09_已知限制.md)

## Development Checks

```bash
pip install -r requirements-dev.txt
python -m ruff check src scripts tests
python -m compileall -q src scripts tests
python scripts/verify_app_smoke.py
python scripts/verify_stability_fixes.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
python scripts/check_release.py
```

## Acknowledgements and License

This project is based on [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) 4.1 and relies on work including VITS, ContentVec, RMVPE, HiFi-GAN, and DiffSinger. Thank you to the original project and all related open-source authors.

Licensed under AGPL-3.0. See [LICENSE](./LICENSE).
