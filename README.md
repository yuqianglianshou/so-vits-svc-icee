# So-VITS-SVC Icee

[English](./README_en.md) | 简体中文

[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](./LICENSE)

一个面向本地离线使用的 So-VITS-SVC 可视化训练与歌声转换工作台。项目基于 so-vits-svc 4.1，重点改善单说话人模型的训练、依赖准备、模型管理和推理体验。

> 当前版本处于发布候选阶段。模型核心和文件格式保持对 so-vits-svc 4.1 的兼容，主要开发路线是 Windows + NVIDIA GPU。

## 使用声明

本项目仅供学习、研究和合法授权场景使用。使用者必须自行确认训练数据、输入音频、模型发布和合成音频发布的授权，不得用于违法、侵权、冒充真人、骚扰、欺诈或政治宗教操纵。

仓库不提供训练完成的角色或真人模型，也不对使用者训练的模型和生成内容负责。继续使用即表示你理解并承担数据及生成内容的相关责任。

## 相对原版的主要变化

| 能力 | so-vits-svc 4.1 | 当前项目 |
| --- | --- | --- |
| 训练入口 | 命令行和旧版 WebUI | 工作区式可视化训练页 |
| 推理流程 | 参数和文件较分散 | 模型导入、质量预设和输出管理 |
| 依赖准备 | 主要依靠手动放置 | 页面检测、自动获取和手动导入引导 |
| 数据组织 | 多个全局目录 | 每个模型使用独立工作区 |
| 目标用户 | 偏向熟悉项目结构的开发者 | 偏向本地 Windows 用户 |
| 模型兼容 | so-vits-svc 4.1 | 保持 4.1 核心与模型兼容路线 |

项目没有宣称发明新的 SVC 模型架构。当前优势主要来自可视化流程、目录整理、依赖管理和更明确的默认使用路线。

## 界面预览

训练页：

![训练页](./images/训练页.png)

推理页：

![推理页](./images/推理页.png)

## 当前支持范围

重点支持：

- Windows 10/11 64 位。
- NVIDIA GPU 和 CUDA 环境。
- Python 3.11。
- 单说话人、单模型工作区训练。
- 本地离线歌声转换。
- `vec768l12` ContentVec、RMVPE 和 44.1 kHz 主路线。

非当前主线：

- Linux 尚未作为正式发布平台完成完整验证。
- macOS 适合页面和轻量检查，不建议用于正式训练。
- 多说话人、实时转换和 ONNX 不属于当前主要支持流程。
- 不承诺特定显卡上的最低显存、训练耗时或固定音质提升幅度。

完整说明见[已知限制](./docs/09_已知限制.md)。

## 快速安装

安装前请准备 Git、Python 3.11、FFmpeg 和可用的 NVIDIA 驱动。以下命令在项目根目录执行：

```powershell
python -m venv .venv311
.venv311\Scripts\activate
python -m pip install --upgrade pip
```

先根据自己的 CUDA 环境安装 PyTorch 和 torchaudio，再安装项目依赖：

```powershell
pip install -r requirements.txt
```

不同 PyTorch/CUDA 组合请以 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)为准。项目的支持边界和已验证组合记录在[环境与兼容性](./docs/07_环境与兼容性.md)。

## 启动

推荐双击：

- `launchers/启动训练界面.bat`
- `launchers/启动推理界面.bat`
- `launchers/启动tensorboard.bat`

也可以在已激活的虚拟环境中运行：

```powershell
python -m src.app_train
python -m src.app_infer
```

## 最短训练流程

1. 启动训练页。
2. 在“训练前依赖与底模”中补齐当前缺失项。
3. 新建一个训练模型工作区。
4. 导入已经获得合法授权的干净 WAV 音频。
5. 执行第 1 步重采样。
6. 执行第 2 步生成配置和文件列表。
7. 执行第 3 步提取 ContentVec、F0 和训练特征。
8. 执行第 4 步主模型训练。
9. 按需执行第 5 步扩散训练和第 6 步音色增强索引。
10. 在推理页导入工作区模型并完成试听。

第一次使用请按[快速开始](./docs/00_快速开始.md)操作；完整流程见[Windows 安装、训练与推理指南](./docs/01_Windows安装训练推理指南.md)。

## 模型依赖

训练页会检查并引导准备以下资产：

- ContentVec HF：`config.json` 和 `model.safetensors`。
- RMVPE：`rmvpe.pt`。
- So-VITS 主模型底模：`G_0.pth` 和 `D_0.pth`。
- 浅扩散底模：`model_0.pt`。
- NSF-HiFiGAN：`model` 和 `config.json`。

建议优先使用训练页的自动获取功能；失败后再按页面链接手动下载和导入。用途、目标路径和来源见[模型依赖清单](./docs/08_模型依赖清单.md)。

## 文档

- [文档索引](./docs/README.md)
- [快速开始](./docs/00_快速开始.md)
- [Windows 安装、训练与推理指南](./docs/01_Windows安装训练推理指南.md)
- [Windows 常见问题排查](./docs/02_Windows常见问题排查.md)
- [核心模型与底模说明](./docs/03_核心模型与底模说明.md)
- [训练流程与原理说明](./docs/04_训练流程与原理说明.md)
- [音质配置推荐](./docs/05_音质配置推荐.md)
- [训练术语与参数速查](./docs/06_训练术语与参数速查.md)

## 开发检查

安装开发依赖后运行：

```bash
pip install -r requirements-dev.txt
python -m ruff check src scripts tests
python -m compileall -q src scripts tests
python scripts/verify_app_smoke.py
python scripts/verify_stability_fixes.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
python scripts/check_release.py
```

详细发布流程见[开发与发布检查](./docs/90_开发与发布检查.md)。

## 致谢与引用

本项目基于 [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) 4.1 开发，并继续使用或参考 VITS、ContentVec、RMVPE、HiFi-GAN 和 DiffSinger 等工作。感谢原项目和所有相关开源作者。

| 名称 | 论文或项目 |
| --- | --- |
| VITS | [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) |
| ContentVec | [ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers](https://arxiv.org/abs/2204.09224) |
| RMVPE | [RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music](https://arxiv.org/abs/2306.15412) |
| HiFi-GAN | [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) |
| DiffSinger | [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446) |

## License

本项目沿用 AGPL-3.0 协议，详见 [LICENSE](./LICENSE)。
