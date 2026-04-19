# SoftVC VITS Singing Voice Conversion

[English](./README_en.md) | [中文简体](./README.md)

[![LICENSE](https://img.shields.io/badge/LICENSE-AGPL3.0-green.svg?style=for-the-badge)](./LICENSE)

这是一个本地离线的 So-VITS-SVC 改造版，当前目标是让用户通过可视化页面完成单说话人训练和高质量歌声音色转换。

## 使用声明

本项目仅供学习、研究和合法授权场景使用。

使用者需要自行确认训练数据、输入音频、模型发布和合成音频发布的授权。本仓库不提供任何模型，也不对用户训练模型或合成音频的用途负责。

禁止使用本项目从事违法、侵权、冒充真人、政治宗教操纵、骚扰欺诈等用途。继续使用即表示你理解并承担由数据和生成内容带来的全部责任。

本项目是基于 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc/) 改造的一个分支，其项目的理论基础与原理并未修改，在此，我向原团队致以最高的敬意。

## 当前主线

当前仓库已经收敛为两个主要入口：

```bash
python -m src.app_train
python -m src.app_infer
```

核心路线：

- 训练方式：单说话人 / 单模型工作区
- 内容编码器：`vec768l12`
- ContentVec 实现：Transformers / HF ContentVec
- F0 预测器：`rmvpe`
- 训练特征包：`*.train.pt`
- 推理目标：离线高质量歌声转换


## 推荐环境

优先推荐：

- Windows 10/11
- NVIDIA GPU + CUDA
- Python 3.11
- 项目根目录 `.venv311` 虚拟环境

说明：

- Windows + NVIDIA GPU 是当前重点验证路线。
- Linux + NVIDIA GPU 理论上可用，但请自行验证。
- macOS 更适合页面、推理或轻量检查，不建议作为正式训练平台。

## 快速安装

```powershell
git clone <你的仓库地址>
cd so-vits-svc-icee
py -3.11 -m venv .venv311
.venv311\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

验证 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

如果输出 `True` 和显卡名称，说明 GPU 环境基本可用。

## 启动页面

训练页：

```powershell
python -m src.app_train
```

推理页：

```powershell
python -m src.app_infer
```

Windows 也可以双击：

```text
launchers/启动训练界面.bat
launchers/启动推理界面.bat
```

训练页负责：

- 检查训练前依赖与底模
- 自动获取或导入依赖文件
- 创建和切换模型工作区
- 导入训练语音
- 执行训练 1-6 步
- 打开 TensorBoard
- 进入推理页

推理页负责：

- 从训练工作区或已导入模型加载模型
- 加载可选扩散模型和音色增强文件
- 切换质量模式
- 转换输入音频
- 导出运行摘要

## 训练前依赖

训练页会检查并引导补齐这些文件：

```text
model_assets/dependencies/encoders/contentvec_hf/config.json
model_assets/dependencies/encoders/contentvec_hf/model.safetensors
model_assets/dependencies/encoders/rmvpe.pt
model_assets/dependencies/base_models/44k/G_0.pth
model_assets/dependencies/base_models/44k/D_0.pth
model_assets/dependencies/base_models/44k/diffusion/model_0.pt
model_assets/dependencies/vocoders/nsf_hifigan/
```

ContentVec HF 必须同时有 `config.json` 和 `model.safetensors`。训练页会显示具体缺哪个文件。

推荐优先在训练页中点击“自动获取当前依赖”；失败时再按页面链接手动下载并导入。

## 目录说明

```text
src/                         # 代码
config_templates/            # 配置模板
training_data/source/        # 原始训练音频
training_data/processed/     # 处理后训练数据
inference_data/inputs/       # 推理输入
inference_data/outputs/      # 推理输出
model_assets/dependencies/   # 训练前依赖与底模
model_assets/workspaces/     # 训练产物工作区
model_assets/imported_models/# 已导入推理模型
docs/                        # 文档
```

推荐一个说话人对应一个模型工作区：

```text
training_data/source/paimeng/
training_data/processed/44k/paimeng/
model_assets/workspaces/paimeng/
```

## 文档

文档入口：

- [文档索引](./docs/README.md)

常用文档：

- [Windows 安装、训练与推理指南](./docs/01_Windows安装训练推理指南.md)
- [Windows 常见问题排查](./docs/02_Windows常见问题排查.md)
- [核心模型与底模说明](./docs/03_核心模型与底模说明.md)
- [训练流程与原理说明](./docs/04_训练流程与原理说明.md)
- [音质配置推荐](./docs/05_音质配置推荐.md)
- [训练术语与参数速查](./docs/06_训练术语与参数速查.md)

## 开发自检

修改训练页、推理页、任务流程或稳定性逻辑后，建议运行：

```bash
python3 -m py_compile src/app_train.py src/app_infer.py
python3 scripts/verify_app_smoke.py
python3 scripts/verify_stability_fixes.py
```

## 引用

| 名称 | 论文 / 项目 |
| --- | --- |
| VITS | [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) |
| ContentVec | [ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers](https://arxiv.org/abs/2204.09224) |
| RMVPE | [RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music](https://arxiv.org/abs/2306.15412v2) |
| HiFi-GAN | [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) |
| DiffSinger / Shallow Diffusion | [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446v3) |

## License

本项目沿用 AGPL-3.0 协议，详见 [LICENSE](./LICENSE)。
