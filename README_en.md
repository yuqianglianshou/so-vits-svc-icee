# SoftVC VITS Singing Voice Conversion

[English](./README_en.md) | [中文简体](./README.md)

[![LICENSE](https://img.shields.io/badge/LICENSE-AGPL3.0-green.svg?style=for-the-badge)](./LICENSE)

This is a local, offline customized So-VITS-SVC repository. The current goal is to let users train single-speaker models and run high-quality singing voice conversion through visual app pages.

## Usage Notice

This project is intended only for learning, research, and legally authorized use cases.

Users are responsible for verifying the authorization of training data, input audio, model distribution, and generated audio. This repository does not provide any model files and is not responsible for how users train models or use generated audio.

Do not use this project for illegal, infringing, impersonation, political or religious manipulation, harassment, fraud, or other harmful purposes. By continuing to use this project, you understand and accept full responsibility for your data and generated content.

This project is a customized fork based on [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc/). Its theoretical foundation and core principles have not been changed. I would like to express my highest respect to the original team.

## Current Mainline

The repository now has two primary entrypoints:

```bash
python -m src.app_train
python -m src.app_infer
```

Current technical route:

- Training mode: single speaker / single model workspace
- Content encoder: `vec768l12`
- ContentVec implementation: Transformers / HF ContentVec
- F0 predictor: `rmvpe`
- Training feature package: `*.train.pt`
- Inference target: offline high-quality singing voice conversion


## Recommended Environment

Recommended:

- Windows 10/11
- NVIDIA GPU + CUDA
- Python 3.11
- `.venv311` virtual environment at the repository root

Notes:

- Windows + NVIDIA GPU is the primary validation route.
- Linux + NVIDIA GPU should be workable, but please validate it yourself.
- macOS is better suited for page checks, inference attempts, or lightweight validation. It is not recommended as the formal training platform.

## Quick Install

```powershell
git clone <your-repository-url>
cd so-vits-svc-icee
py -3.11 -m venv .venv311
.venv311\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If it prints `True` and your GPU name, the GPU environment is basically ready.

## Launch Apps

Training page:

```powershell
python -m src.app_train
```

Inference page:

```powershell
python -m src.app_infer
```

On Windows, you can also double-click:

```text
launchers/启动训练界面.bat
launchers/启动推理界面.bat
```

The training page handles:

- checking training dependencies and base models
- automatically downloading or importing dependency files
- creating and switching model workspaces
- importing training audio
- running training steps 1-6
- opening TensorBoard
- opening the inference page

The inference page handles:

- loading models from training workspaces or imported models
- loading optional diffusion models and timbre-enhancement files
- switching quality modes
- converting input audio
- exporting runtime summaries

## Training Dependencies

The training page checks and guides you to prepare these files:

```text
model_assets/dependencies/encoders/contentvec_hf/config.json
model_assets/dependencies/encoders/contentvec_hf/model.safetensors
model_assets/dependencies/encoders/rmvpe.pt
model_assets/dependencies/base_models/44k/G_0.pth
model_assets/dependencies/base_models/44k/D_0.pth
model_assets/dependencies/base_models/44k/diffusion/model_0.pt
model_assets/dependencies/vocoders/nsf_hifigan/
```

ContentVec HF requires both `config.json` and `model.safetensors`. The training page shows exactly which file is missing.

Prefer clicking "auto fetch current dependency" in the training page. If that fails, use the page-provided links to download manually and import the files.

## Directory Layout

```text
src/                         # source code
config_templates/            # config templates
training_data/source/        # raw training audio
training_data/processed/     # preprocessed training data
inference_data/inputs/       # inference inputs
inference_data/outputs/      # inference outputs
model_assets/dependencies/   # training dependencies and base models
model_assets/workspaces/     # training workspaces and outputs
model_assets/imported_models/# imported inference models
docs/                        # documentation
```

Recommended mapping: one speaker corresponds to one model workspace.

```text
training_data/source/paimeng/
training_data/processed/44k/paimeng/
model_assets/workspaces/paimeng/
```

## Documentation

Documentation index:

- [Documentation Index](./docs/README.md)

Common documents:

- [Windows Install, Training, and Inference Guide](./docs/01_Windows安装训练推理指南.md)
- [Windows Troubleshooting](./docs/02_Windows常见问题排查.md)
- [Core Models and Base Models](./docs/03_核心模型与底模说明.md)
- [Training Flow and Theory](./docs/04_训练流程与原理说明.md)
- [Audio Quality Configuration Recommendations](./docs/05_音质配置推荐.md)
- [Training Terms and Parameter Quick Reference](./docs/06_训练术语与参数速查.md)

## Development Checks

After changing the training page, inference page, task flow, or stability-related logic, run:

```bash
python3 -m py_compile src/app_train.py src/app_infer.py
python3 scripts/verify_app_smoke.py
python3 scripts/verify_stability_fixes.py
```

## References

| Name | Paper / Project |
| --- | --- |
| VITS | [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) |
| ContentVec | [ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers](https://arxiv.org/abs/2204.09224) |
| RMVPE | [RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music](https://arxiv.org/abs/2306.15412v2) |
| HiFi-GAN | [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) |
| DiffSinger / Shallow Diffusion | [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446v3) |

## License

This project follows the AGPL-3.0 license. See [LICENSE](./LICENSE).
