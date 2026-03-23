<div align="center">
<img alt="LOGO" src="https://avatars.githubusercontent.com/u/127122328?s=400&u=5395a98a4f945a3a50cb0cc96c2747505d190dbc&v=4" width="300" height="300" />

# SoftVC VITS Singing Voice Conversion

[**English**](./README.md) | [**中文简体**](./README_zh_CN.md)

[![在Google Cloab中打开](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/svc-develop-team/so-vits-svc/blob/4.1-Stable/sovits4_for_colab.ipynb)
[![LICENSE](https://img.shields.io/badge/LICENSE-AGPL3.0-green.svg?style=for-the-badge)](https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/LICENSE)

本轮限时更新即将结束，仓库将进入Archieve状态，望周知

</div>


#### ✨ 带有 F0 曲线编辑器，角色混合时间轴编辑器的推理端 (Onnx 模型的用途）: [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio)

#### ✨ 改善了交互的一个分支推荐: [34j/so-vits-svc-fork](https://github.com/34j/so-vits-svc-fork)

#### ✨ 支持实时转换的一个客户端: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

**本项目与 Vits 有着根本上的不同。Vits 是 TTS，本项目是 SVC。本项目无法实现 TTS，Vits 也无法实现 SVC，这两个项目的模型是完全不通用的。**

## 重要通知

这个项目是为了让开发者最喜欢的动画角色唱歌而开发的，任何涉及真人的东西都与开发者的意图背道而驰。

## 声明

本项目为开源、离线的项目，SvcDevelopTeam 的所有成员与本项目的所有开发者以及维护者（以下简称贡献者）对本项目没有控制力。本项目的贡献者从未向任何组织或个人提供包括但不限于数据集提取、数据集加工、算力支持、训练支持、推理等一切形式的帮助；本项目的贡献者不知晓也无法知晓使用者使用该项目的用途。故一切基于本项目训练的 AI 模型和合成的音频都与本项目贡献者无关。一切由此造成的问题由使用者自行承担。

此项目完全离线运行，不能收集任何用户信息或获取用户输入数据。因此，这个项目的贡献者不知道所有的用户输入和模型，因此不负责任何用户输入。

本项目只是一个框架项目，本身并没有语音合成的功能，所有的功能都需要用户自己训练模型。同时，这个项目没有任何模型，任何二次分发的项目都与这个项目的贡献者无关。

## 📏 使用规约

# Warning：请自行解决数据集授权问题，禁止使用非授权数据集进行训练！任何由于使用非授权数据集进行训练造成的问题，需自行承担全部责任和后果！与仓库、仓库维护者、svc develop team 无关！

1. 本项目是基于学术交流目的建立，仅供交流与学习使用，并非为生产环境准备。
2. 任何发布到视频平台的基于 sovits 制作的视频，都必须要在简介明确指明用于变声器转换的输入源歌声、音频，例如：使用他人发布的视频 / 音频，通过分离的人声作为输入源进行转换的，必须要给出明确的原视频、音乐链接；若使用是自己的人声，或是使用其他歌声合成引擎合成的声音作为输入源进行转换的，也必须在简介加以说明。
3. 由输入源造成的侵权问题需自行承担全部责任和一切后果。使用其他商用歌声合成软件作为输入源时，请确保遵守该软件的使用条例，注意，许多歌声合成引擎使用条例中明确指明不可用于输入源进行转换！
4. 禁止使用该项目从事违法行为与宗教、政治等活动，该项目维护者坚决抵制上述行为，不同意此条则禁止使用该项目。
5. 继续使用视为已同意本仓库 README 所述相关条例，本仓库 README 已进行劝导义务，不对后续可能存在问题负责。
6. 如果将此项目用于任何其他企划，请提前联系并告知本仓库作者，十分感谢。

## 📝 模型简介

歌声音色转换模型，通过 SoftVC 内容编码器提取源音频语音特征，与 F0 同时输入 VITS 替换原本的文本输入达到歌声转换的效果。同时，更换声码器为 [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) 解决断音问题。

### 🆕 4.1-Stable 版本更新内容

+ 特征输入更换为 [Content Vec](https://github.com/auspicious3000/contentvec) 的第 12 层 Transformer 输出，并兼容 4.0 分支
+ 更新浅层扩散，可以使用浅层扩散模型提升音质
+ 增加 whisper 语音编码器的支持
+ 增加静态/动态声线融合
+ 增加响度嵌入
+ 增加特征检索，来自于 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

### 🆕 关于兼容 4.0 模型的问题

+ 可通过修改 4.0 模型的 config.json 对 4.0 的模型进行支持，需要在 config.json 的 model 字段中添加 speech_encoder 字段，具体见下

```
  "model": {
    .........
    "ssl_dim": 256,
    "n_speakers": 200,
    "speech_encoder":"vec256l9"
  }
```

### 🆕 关于浅扩散
![Diagram](shadowdiffusion.png)

## 💬 关于 Python 版本问题

根据当前这套改造版仓库的实际验证结果，稳定环境基线是 `Python 3.11.x`

## 🧪 环境约定

当前项目统一使用 Python 自带的 `venv` 作为默认环境方案：

- 虚拟环境固定创建在项目根目录下的 `.venv311`
- 不再默认使用 Conda / Anaconda
- Windows 批处理脚本也默认从 `.venv311\Scripts\python.exe` 启动

标准初始化方式：

```shell
python -m venv .venv311
```

## ✅ 提交前自检

如果你修改了入口页面、训练流程、推理流程或稳定性相关逻辑，建议提交前至少运行下面两条检查：

```shell
python3 scripts/verify_stability_fixes.py
python3 scripts/verify_app_smoke.py
```

作用说明：

- `verify_stability_fixes.py`
  - 检查近期修过的稳定性问题是否回退
  - 包括推理解包、说话人解析、DDP sampler、预处理并行逻辑

- `verify_app_smoke.py`
  - 检查当前两个入口页面的关键引用和流程约束是否仍然存在
  - 包括：
    - `app_train.py` 的训练前依赖与底模、数据集安全与一键流程
    - `app_infer.py` 的质量模式与运行摘要

## 🚀 当前仓库推荐工作流

如果你使用的是当前这个仓库改造版，推荐优先按下面的方式使用，而不是直接从底层脚本开始。

### 1. 创建虚拟环境并安装依赖

```shell
python -m venv .venv311
```

```shell
# Windows
.venv311\Scripts\activate

# macOS / Linux
source .venv311/bin/activate

pip install --upgrade pip setuptools wheel
# Windows + NVIDIA GPU（推荐）
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. 训练模型

推荐入口：

```shell
python app_train.py
```

Windows 也可以直接双击：

- `启动训练界面.bat`

训练页顶部负责：

- 检查训练前依赖与底模
- 导入缺失文件到标准目录
- 补齐 `pretrain/encoders/`、`pretrain/vocoders/` 和 `pretrain/base_models/44k/` 下的训练依赖与底模

当前训练页支持：

- 导入 `dataset_raw/<说话人目录>/*.wav`
- 自动建议目录名
- 删除当前说话人目录
- 自动同步处理后目录 `dataset/44k/<说话人目录>`
- 默认把“模型输出名”同步成当前说话人目录名
- 一键执行重采样、配置生成、特征提取、主模型训练
- 查看训练前检查、阶段判断、任务状态、日志和错误提示

当前推荐按 **单说话人模式** 使用：

- 一个说话人目录对应一轮训练
- 一个说话人建议对应一个独立模型输出目录 `logs/<模型名>/`
- 如果你要训练第二个说话人，不是把两个说话人混进一个模型，而是切换说话人目录并使用新的模型输出名

### 3. 歌声转换

推荐入口：

```shell
python app_infer.py
```

当前推理页支持：

- 加载主模型、音质增强模型、音色增强文件
- 高音质质量模式切换
- 运行摘要导出
- 面向离线高质量歌声转换

### 4. 什么时候看下面的旧章节

下面保留的 `数据预处理 / 训练 / 推理` 章节，主要用于：

- 直接调用底层脚本
- 理解底层训练链路
- 在 Windows + GPU 机器上做命令行排障

如果你只是正常使用当前仓库，优先使用：

- `app_train.py`
- `app_infer.py`

## 📥 预先下载的模型文件

#### **必须项**

**以下编码器需要选择一个使用**

##### **1. 若使用 contentvec 作为声音编码器（推荐）**

`vec768l12` 当前固定使用 Transformers / HF ContentVec 路线。

+ ContentVec HF 目录：`pretrain/encoders/contentvec_hf/`
  + 目录内需要包含：
    + `config.json`
    + `model.safetensors`
  + 当前项目锁定来源：
    + `lengyue233/content-vec-best`

如果本地没有这个目录，程序可以回退到锁定的 Hugging Face 来源自动获取；为了长期稳定，仍建议优先保留本地目录。

##### **2. 若使用 hubertsoft 作为声音编码器**
+ soft vc hubert：[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  + 放在`pretrain`目录下

##### **3. 若使用 Whisper-ppg 作为声音编码器**
+ 下载模型 [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 该模型适配`whisper-ppg`
+ 下载模型 [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), 该模型适配`whisper-ppg-large`
  + 放在`pretrain`目录下

##### **4. 若使用 cnhubertlarge 作为声音编码器**
+ 下载模型 [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)
  + 放在`pretrain`目录下

##### **5. 若使用 dphubert 作为声音编码器**
+ 下载模型 [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)
  + 放在`pretrain`目录下

##### **6. 若使用 WavLM 作为声音编码器**
+ 下载模型 [WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), 该模型适配`wavlmbase+`
  + 放在`pretrain`目录下

##### **7. 若使用 OnnxHubert/ContentVec 作为声音编码器**
+ 下载模型 [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)
  + 放在`pretrain`目录下

#### **编码器列表**
- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"
- "wavlmbase+"

#### **可选项（强烈建议使用）**

+ 预训练底模文件：
  + [G_0.pth（vec768l12）](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec768l12/G_0.pth)
  + [D_0.pth（vec768l12）](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec768l12/D_0.pth)
  + 放在`pretrain/base_models/44k`目录下

+ 扩散模型预训练底模文件：
  + [model_0.pt（diffusion/768l12）](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/diffusion/768l12/model_0.pt)
  + 放在`pretrain/base_models/44k/diffusion`目录下

训练开始时，程序会自动把这组底模同步到运行期实验目录 `logs/44k/`，所以 `logs/44k` 现在主要表示训练产物目录，而不是长期存放底模的目录。

上面这组链接对应当前仓库默认的 `vec768l12` / `ContentVec` 路线，不要和 `vec256l9` 等其他目录里的同名底模混用。

虽然底模一般不会引起什么版权问题，但还是请注意一下，比如事先询问作者，又或者作者在模型描述中明确写明了可行的用途

#### **可选项（根据情况选择）**

##### NSF-HIFIGAN

如果使用`NSF-HIFIGAN 增强器`或`浅层扩散`的话，需要下载预训练的 NSF-HIFIGAN 模型，如果不需要可以不下载

+ 预训练的 NSF-HIFIGAN 声码器 ：[nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)
  + 解压后，将四个文件放在`pretrain/vocoders/nsf_hifigan`目录下

```shell
# nsf_hifigan
wget -P pretrain/ https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
unzip -od pretrain/vocoders/nsf_hifigan pretrain/nsf_hifigan_20221211.zip
# 也可手动下载放在 pretrain/vocoders/nsf_hifigan 目录
# 地址：https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1
```

##### RMVPE

如果使用`rmvpe`F0预测器的话，需要下载预训练的 RMVPE 模型

+ 下载模型[rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip)，目前首推该权重。
  + 解压缩`rmvpe.zip`，并将其中的`model.pt`文件改名为`rmvpe.pt`并放在`pretrain/encoders`目录下

+ ~~下载模型 [rmvpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt)~~
  + ~~放在`pretrain/encoders`目录下~~

##### FCPE(预览版)

> 你说的对,但是[FCPE](https://github.com/CNChTu/MelPE)是由svc-develop-team自主研发的一款全新的F0预测器，后面忘了

[FCPE(Fast Context-base Pitch Estimator)](https://github.com/CNChTu/MelPE)是一个为实时语音转换所设计的专用F0预测器，他将在未来成为Sovits实时语音转换的首选F0预测器.（论文未来会有的）

如果使用 `fcpe` F0预测器的话，需要下载预训练的 FCPE 模型

+ 下载模型 [fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)
  + 放在`pretrain`目录下


## 📊 数据集准备（底层脚本参考）

当前仓库推荐直接在 `app_train.py` 页面中导入说话人目录。  
如果你走命令行底层流程，数据集结构仍然是下面这种：

```
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
对于每一个音频文件的名称并没有格式的限制(`000001.wav`~`999999.wav`之类的命名方式也是合法的)，不过文件类型必须是`wav`。

可以自定义说话人名称

```
dataset_raw
└───suijiSUI
    ├───1.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

## 🛠️ 数据预处理（底层脚本参考）

### 0. 音频切片

将音频切片至`5s - 15s`, 稍微长点也无伤大雅，实在太长可能会导致训练中途甚至预处理就爆显存

可以使用 [audio-slicer-GUI](https://github.com/flutydeer/audio-slicer)、[audio-slicer-CLI](https://github.com/openvpi/audio-slicer)

一般情况下只需调整其中的`Minimum Interval`，普通陈述素材通常保持默认即可，歌唱素材可以调整至`100`甚至`50`

切完之后手动删除过长过短的音频

**如果你使用 Whisper-ppg 声音编码器进行训练，所有的切片长度必须小于 30s**

### 1. 重采样至 44100Hz 单声道

```shell
python -m train_pipeline.resample
```

#### 注意

当前 `train_pipeline/resample.py` 只负责裁剪静音和重采样，不再在项目内做响度、电平或限幅处理。训练音频的响度整理、增益控制和 limiter，请在进入项目之前由你自己的音频处理流程完成。

### 2. 自动划分训练集、验证集，以及自动生成配置文件

```shell
python -m train_pipeline.preprocess_flist_config --speech_encoder vec768l12
```

speech_encoder 拥有以下选择

```
vec768l12
vec256l9
hubertsoft
whisper-ppg
whisper-ppg-large
cnhubertlarge
dphubert
wavlmbase+
```

如果省略 speech_encoder 参数，默认值为 vec768l12

**使用响度嵌入**

若使用响度嵌入，需要增加`--vol_aug`参数，比如：

```shell
python -m train_pipeline.preprocess_flist_config --speech_encoder vec768l12 --vol_aug
```
使用后训练出的模型将匹配到输入源响度，否则为训练集响度。

#### 此时可以在生成的 config.json 与 diffusion.yaml 修改部分参数

##### config.json

* `keep_ckpts`：训练时保留最后几个模型，`0`为保留所有，默认只保留最后`3`个

* `all_in_mem`：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 **远大于** 数据集体积时可以启用

* `batch_size`：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可

* `vocoder_name` : 选择一种声码器，默认为`nsf-hifigan`.

##### diffusion.yaml

* `cache_all_data`：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 **远大于** 数据集体积时可以启用

* `duration`：训练时音频切片时长，可根据显存大小调整，**注意，该值必须小于训练集内音频的最短时间！**

* `batch_size`：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可

* `timesteps` : 扩散模型总步数，默认为 1000.

* `k_step_max` : 训练时可仅训练`k_step_max`步扩散以节约训练时间，注意，该值必须小于`timesteps`，0 为训练整个扩散模型，**注意，如果不训练整个扩散模型将无法使用仅扩散模型推理！**

##### **声码器列表**

```
nsf-hifigan
nsf-snake-hifigan
```

### 3. 生成 hubert 与 f0

```shell
python -m train_pipeline.preprocess_hubert_f0 --f0_predictor dio
```

f0_predictor 拥有以下选择

```
crepe
dio
pm
harvest
rmvpe
fcpe
```

如果训练集过于嘈杂，请使用 crepe 处理 f0

如果省略 f0_predictor 参数，默认值为 rmvpe

尚若需要浅扩散功能（可选），需要增加--use_diff 参数，比如

```shell
python -m train_pipeline.preprocess_hubert_f0 --f0_predictor dio --use_diff
```

**加速预处理**
如若您的数据集比较大，可以尝试添加`--num_processes`参数：
```shell
python -m train_pipeline.preprocess_hubert_f0 --f0_predictor dio --use_diff --num_processes 8
```
所有的Workers会被自动分配到多个线程上

执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除 dataset_raw 文件夹了

## 🏋️‍ 训练（底层脚本参考）

### 主模型训练

```shell
python -m train_pipeline.train -c configs/config.json -m 44k
```

### 扩散模型（可选）

尚若需要浅扩散功能，需要训练扩散模型，扩散模型训练方法为：

```shell
python -m train_pipeline.train_diff -c configs/diffusion.yaml
```

模型训练结束后，模型文件保存在`logs/44k`目录下，扩散模型在`logs/44k/diffusion`下

## 🤖 推理（底层脚本参考）

如果你使用当前仓库改造版，优先使用 [app_infer.py](app_infer.py)。  
下面这部分是 [services/inference_main.py](services/inference_main.py) 的命令行参考。

```shell
# 例
python -m services.inference_main -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```

必填项部分：
+ `-m` | `--model_path`：模型路径
+ `-c` | `--config_path`：配置文件路径
+ `-n` | `--clean_names`：wav 文件名列表，放在 raw 文件夹下
+ `-t` | `--trans`：音高调整，支持正负（半音）
+ `-s` | `--spk_list`：合成目标说话人名称
+ `-cl` | `--clip`：音频强制切片，默认 0 为自动切片，单位为秒/s

可选项部分：部分具体见下一节
+ `-lg` | `--linear_gradient`：两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值 0，单位为秒
+ `-f0p` | `--f0_predictor`：选择 F0 预测器，可选择 crepe,pm,dio,harvest,rmvpe,fcpe, 默认为 pm（注意：crepe 为原 F0 使用均值滤波器）
+ `-a` | `--auto_predict_f0`：语音转换自动预测音高，转换歌声时不要打开这个会严重跑调
+ `-cm` | `--cluster_model_path`：聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填
+ `-cr` | `--cluster_infer_ratio`：聚类方案或特征检索占比，范围 0-1，若没有训练聚类模型或特征检索则默认 0 即可
+ `-eh` | `--enhance`：是否使用 NSF_HIFIGAN 增强器，该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭
+ `-shd` | `--shallow_diffusion`：是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN 增强器将会被禁止
+ `-usm` | `--use_spk_mix`：是否使用角色融合/动态声线融合
+ `-lea` | `--loudness_envelope_adjustment`：输入源响度包络替换输出响度包络融合比例，越靠近 1 越使用输出响度包络
+ `-fr` | `--feature_retrieval`：是否使用特征检索，如果使用聚类模型将被禁用，且 cm 与 cr 参数将会变成特征检索的索引路径与混合比例

浅扩散设置：
+ `-dm` | `--diffusion_model_path`：扩散模型路径
+ `-dc` | `--diffusion_config_path`：扩散模型配置文件路径
+ `-ks` | `--k_step`：扩散步数，越大越接近扩散模型的结果，默认 100
+ `-od` | `--only_diffusion`：纯扩散模式，该模式不会加载 sovits 模型，以扩散模型推理
+ `-se` | `--second_encoding`：二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差

### 注意！

如果使用`whisper-ppg` 声音编码器进行推理，需要将`--clip`设置为 25，`-lg`设置为 1。否则将无法正常推理。

## 🤔 可选项

如果前面的效果已经满意，或者没看明白下面在讲啥，那后面的内容都可以忽略，不影响模型使用（这些可选项影响比较小，可能在某些特定数据上有点效果，但大部分情况似乎都感知不太明显）

### 自动 f0 预测

4.0 模型训练过程会训练一个 f0 预测器，对于语音转换可以开启自动音高预测，如果效果不好也可以使用手动的，但转换歌声时请不要启用此功能！！！会严重跑调！！
+ 在 inference_main 中设置 auto_predict_f0 为 true 即可

### 聚类音色泄漏控制

介绍：聚类方案可以减小音色泄漏，使得模型训练出来更像目标的音色（但其实不是特别明显），但是单纯的聚类方案会降低模型的咬字（会口齿不清）（这个很明显），本模型采用了融合的方式，可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点

使用聚类前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型，虽然效果比较有限，但训练成本也比较低

+ 训练过程：
  + 使用 cpu 性能较好的机器训练，据我的经验在腾讯云 6 核 cpu 训练每个 speaker 需要约 4 分钟即可完成训练
  + 执行`python cluster/train_cluster.py`，模型的输出会在`logs/44k/kmeans_10000.pt`
  + 聚类模型目前可以使用 gpu 进行训练，执行`python cluster/train_cluster.py --gpu`
+ 推理过程：
  + `services/inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/kmeans_10000.pt`
  + `services/inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用聚类，`1`为只使用聚类，通常设置`0.5`即可

### 特征检索

介绍：跟聚类方案一样可以减小音色泄漏，咬字比聚类稍好，但会降低推理速度，采用了融合的方式，可以线性控制特征检索与非特征检索的占比，

+ 训练过程：
  首先需要在生成 hubert 与 f0 后执行：

```shell
python -m train_pipeline.train_index -c configs/config.json
```

模型的输出会在`logs/44k/feature_and_index.pkl`

+ 推理过程：
  + 需要首先指定`--feature_retrieval`，此时聚类方案会自动切换到特征检索方案
  + `services/inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/feature_and_index.pkl`
  + `services/inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用特征检索，`1`为只使用特征检索，通常设置`0.5`即可


## 🗜️ 模型压缩

生成的模型含有继续训练所需的信息。如果确认不再训练，可以移除模型中此部分信息，得到约 1/3 大小的最终模型。

使用 [tools/compress_model.py](tools/compress_model.py)

```shell
# 例
python -m tools.compress_model -c="configs/config.json" -i="logs/44k/G_30400.pth" -o="logs/44k/release.pth"
```

## 👨‍🔧 声线混合

### 静态声线混合

**参考`app_infer.py`文件中，小工具/实验室特性的静态声线融合。**

介绍：该功能可以将多个声音模型合成为一个声音模型（多个模型参数的凸组合或线性组合），从而制造出现实中不存在的声线
**注意：**

1. 该功能仅支持单说话人的模型
2. 如果强行使用多说话人模型，需要保证多个模型的说话人数量相同，这样可以混合同一个 SpaekerID 下的声音
3. 保证所有待混合模型的 config.json 中的 model 字段是相同的
4. 输出的混合模型可以使用待合成模型的任意一个 config.json，但聚类模型将不能使用
5. 批量上传模型的时候最好把模型放到一个文件夹选中后一起上传
6. 混合比例调整建议大小在 0-100 之间，也可以调为其他数字，但在线性组合模式下会出现未知的效果
7. 混合完毕后，文件将会保存在项目根目录中，文件名为 output.pth
8. 凸组合模式会将混合比例执行 Softmax 使混合比例相加为 1，而线性组合模式不会

### 动态声线混合

**参考`spkmix.py`文件中关于动态声线混合的介绍**

角色混合轨道 编写规则：

角色 ID : \[\[起始时间 1, 终止时间 1, 起始数值 1, 起始数值 1], [起始时间 2, 终止时间 2, 起始数值 2, 起始数值 2]]

起始时间和前一个的终止时间必须相同，第一个起始时间必须为 0，最后一个终止时间必须为 1 （时间的范围为 0-1）

全部角色必须填写，不使用的角色填、[\[0., 1., 0., 0.]] 即可

融合数值可以随便填，在指定的时间段内从起始数值线性变化为终止数值，内部会自动确保线性组合为 1（凸组合条件），可以放心使用

推理的时候使用`--use_spk_mix`参数即可启用动态声线混合

## 📤 Onnx 导出

使用 [tools/onnx_export.py](tools/onnx_export.py)

+ 新建文件夹：`checkpoints` 并打开
+ 在`checkpoints`文件夹中新建一个文件夹作为项目文件夹，文件夹名为你的项目名称，比如`aziplayer`
+ 将你的模型更名为`model.pth`，配置文件更名为`config.json`，并放置到刚才创建的`aziplayer`文件夹下
+ 将 [tools/onnx_export.py](tools/onnx_export.py) 中`path = "NyaruTaffy"` 的 `"NyaruTaffy"` 修改为你的项目名称，`path = "aziplayer" (onnx_export_speaker_mix，为支持角色混合的 onnx 导出）`
+ 运行 `python -m tools.onnx_export`
+ 等待执行完毕，在你的项目文件夹下会生成一个`model.onnx`，即为导出的模型

注意：Hubert Onnx 模型请使用 MoeSS 提供的模型，目前无法自行导出（fairseq 中 Hubert 有不少 onnx 不支持的算子和涉及到常量的东西，在导出时会报错或者导出的模型输入输出 shape 和结果都有问题）

## 📎 引用及论文

| URL | 名称 | 标题 | 源码 |
| --- | ----------- | ----- | --------------------- |
|[2106.06103](https://arxiv.org/abs/2106.06103) | VITS (Synthesizer)| Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech | [jaywalnut310/vits](https://github.com/jaywalnut310/vits) |
|[2111.02392](https://arxiv.org/abs/2111.02392) | SoftVC (Speech Encoder)| A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion | [bshall/hubert](https://github.com/bshall/hubert) |
|[2204.09224](https://arxiv.org/abs/2204.09224) | ContentVec (Speech Encoder)| ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) |
|[2212.04356](https://arxiv.org/abs/2212.04356) | Whisper (Speech Encoder) | Robust Speech Recognition via Large-Scale Weak Supervision | [openai/whisper](https://github.com/openai/whisper) |
|[2110.13900](https://arxiv.org/abs/2110.13900) | WavLM (Speech Encoder) | WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | [microsoft/unilm/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm) |
|[2305.17651](https://arxiv.org/abs/2305.17651) | DPHubert (Speech Encoder) | DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models | [pyf98/DPHuBERT](https://github.com/pyf98/DPHuBERT) |
|[DOI:10.21437/Interspeech.2017-68](http://dx.doi.org/10.21437/Interspeech.2017-68) | Harvest (F0 Predictor) | Harvest: A high-performance fundamental frequency estimator from speech signals | [mmorise/World/harvest](https://github.com/mmorise/World/blob/master/src/harvest.cpp) |
|[aes35-000039](https://www.aes.org/e-lib/online/browse.cfm?elib=15165) | Dio (F0 Predictor) | Fast and reliable F0 estimation method based on the period extraction of vocal fold vibration of singing voice and speech | [mmorise/World/dio](https://github.com/mmorise/World/blob/master/src/dio.cpp) |
|[8461329](https://ieeexplore.ieee.org/document/8461329) | Crepe (F0 Predictor) | Crepe: A Convolutional Representation for Pitch Estimation | [maxrmorrison/torchcrepe](https://github.com/maxrmorrison/torchcrepe) |
|[DOI:10.1016/j.wocn.2018.07.001](https://doi.org/10.1016/j.wocn.2018.07.001) | Parselmouth (F0 Predictor) | Introducing Parselmouth: A Python interface to Praat | [YannickJadoul/Parselmouth](https://github.com/YannickJadoul/Parselmouth) |
|[2306.15412v2](https://arxiv.org/abs/2306.15412v2) | RMVPE (F0 Predictor) | RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) |
|[2010.05646](https://arxiv.org/abs/2010.05646) | HIFIGAN (Vocoder) | HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | [jik876/hifi-gan](https://github.com/jik876/hifi-gan) |
|[1810.11946](https://arxiv.org/abs/1810.11946.pdf) | NSF (Vocoder) | Neural source-filter-based waveform model for statistical parametric speech synthesis | [openvpi/DiffSinger/modules/nsf_hifigan](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan)
|[2006.08195](https://arxiv.org/abs/2006.08195) | Snake (Vocoder) | Neural Networks Fail to Learn Periodic Functions and How to Fix It | [EdwardDixon/snake](https://github.com/EdwardDixon/snake)
|[2105.02446v3](https://arxiv.org/abs/2105.02446v3) | Shallow Diffusion (PostProcessing)| DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism | [CNChTu/Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) |
|[K-means](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=01D65490BADCC216F350D06F84D721AD?doi=10.1.1.308.8619&rep=rep1&type=pdf) | Feature K-means Clustering (PreProcessing)| Some methods for classification and analysis of multivariate observations | 本代码库 |
| | Feature TopK Retrieval (PreProcessing)| Retrieval based Voice Conversion | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |

## ☀️ 旧贡献者

因为某些原因原作者进行了删库处理，本仓库重建之初由于组织成员疏忽直接重新上传了所有文件导致以前的 contributors 全部木大，现在在 README 里重新添加一个旧贡献者列表

*某些成员已根据其个人意愿不将其列出*

<table>
  <tr>
    <td align="center"><a href="https://github.com/MistEO"><img src="https://avatars.githubusercontent.com/u/18511905?v=4" width="100px;" alt=""/><br /><sub><b>MistEO</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/XiaoMiku01"><img src="https://avatars.githubusercontent.com/u/54094119?v=4" width="100px;" alt=""/><br /><sub><b>XiaoMiku01</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/ForsakenRei"><img src="https://avatars.githubusercontent.com/u/23041178?v=4" width="100px;" alt=""/><br /><sub><b>しぐれ</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/TomoGaSukunai"><img src="https://avatars.githubusercontent.com/u/25863522?v=4" width="100px;" alt=""/><br /><sub><b>TomoGaSukunai</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Plachtaa"><img src="https://avatars.githubusercontent.com/u/112609742?v=4" width="100px;" alt=""/><br /><sub><b>Plachtaa</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/zdxiaoda"><img src="https://avatars.githubusercontent.com/u/45501959?v=4" width="100px;" alt=""/><br /><sub><b>zd 小达</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Archivoice"><img src="https://avatars.githubusercontent.com/u/107520869?v=4" width="100px;" alt=""/><br /><sub><b>凍聲響世</b></sub></a><br /></td>
  </tr>
</table>

## 📚 一些法律条例参考

#### 任何国家，地区，组织和个人使用此项目必须遵守以下法律

#### 《民法典》

##### 第一千零一十九条

任何组织或者个人不得以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。未经肖像权人同意，不得制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。未经肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。对自然人声音的保护，参照适用肖像权保护的有关规定。

##### 第一千零二十四条

【名誉权】民事主体享有名誉权。任何组织或者个人不得以侮辱、诽谤等方式侵害他人的名誉权。

##### 第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。

#### 《[中华人民共和国宪法](http://www.gov.cn/guoqing/2018-03/22/content_5276318.htm)》

#### 《[中华人民共和国刑法](http://gongbao.court.gov.cn/Details/f8e30d0689b23f57bfc782d21035c3.html?sw=中华人民共和国刑法)》

#### 《[中华人民共和国民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》

#### 《[中华人民共和国合同法](http://www.npc.gov.cn/zgrdw/npc/lfzt/rlyw/2016-07/01/content_1992739.htm)》

## 💪 感谢所有的贡献者
<a href="https://github.com/svc-develop-team/so-vits-svc/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=svc-develop-team/so-vits-svc" />
</a>
