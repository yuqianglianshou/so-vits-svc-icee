# Windows 安装、训练与推理指南

“怎么从零跑起来”：安装环境、启动训练页、补齐依赖、导入数据、训练、打开推理页、做首次验证。

遇到报错或现象异常时，请看：

- [Windows 常见问题排查](02_Windows常见问题排查.md)

## 0. 当前推荐路线

1. 推荐环境：Windows 10/11 + NVIDIA GPU + Python 3.11。
2. 训练主入口：`python -m src.app_train`。
3. 推理主入口：`python -m src.app_infer`。
4. 打开训练页面处理训练前依赖、底模、数据导入和 1-6 步训练。

## 1. 准备 Windows 机器

建议准备：

1. Windows 10/11 64 位。
2. NVIDIA 显卡，显存 >= 8GB。
3. 最新 NVIDIA 驱动。
4. Python 3.11。
5. Git。

打开 PowerShell 检查 Python 3.11 版本是否存在：

```powershell
py -3.11 --version
```

如果机器未安装 Python 3.11：

1. 打开 Python 官网下载页：<https://www.python.org/downloads/windows/>。
2. 下载 Python 3.11.x 的 Windows installer，建议选择 `Windows installer (64-bit)`。
3. 运行安装包，先勾选 `Add python.exe to PATH`，再点击 `Install Now`。
4. 安装完成后，关闭当前 PowerShell，重新打开 PowerShell。
5. 再次检查：

```powershell
py -3.11 --version
```


## 2. 下载项目并安装依赖

在 PowerShell 中执行：

```powershell
# 下载项目代码
git clone https://github.com/yuqianglianshou/so-vits-svc-icee.git

# 进入项目目录
cd so-vits-svc-icee

# 使用 Python 3.11 创建当前项目的虚拟环境
py -3.11 -m venv .venv311

# 激活虚拟环境
.venv311\Scripts\activate

# 升级 pip 和基础安装工具
python -m pip install --upgrade pip setuptools wheel

# 安装支持 CUDA 11.8 的 PyTorch 和 torchaudio
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目所需的其他 Python 依赖
pip install -r requirements.txt
```

安装后验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

看到 `True` 和显卡名称，说明 GPU 环境基本可用。

注意：

1. `.bat` 启动脚本默认使用项目根目录下的 `.venv311`。
2. 主模型训练和扩散训练都按 NVIDIA GPU + CUDA 场景准备。

## 3. 启动训练页

推荐双击：

```text
launchers/启动训练界面.bat
```

或者在已激活虚拟环境的 PowerShell 中运行：

```powershell
python -m src.app_train
```

训练页负责：

- 检查训练前依赖与底模
- 自动获取或导入依赖文件
- 创建和切换模型工作区
- 导入训练语音
- 执行训练 1-6 步
- 打开 TensorBoard
- 进入推理页

如果浏览器没有自动打开，按终端或页面提示手动访问本地地址。

## 4. 补齐训练前依赖与底模

进入训练页后，先打开“训练前依赖与底模”区域。推荐优先用页面里的“自动获取当前依赖”，失败时再手动下载并导入。

当前训练页会检查：

1. ContentVec HF 模型目录：
   - 目录：`model_assets/dependencies/encoders/contentvec_hf/`
   - 必需文件：`config.json` （已经内置）
   - 必需文件：`model.safetensors`
2. RMVPE：
   - `model_assets/dependencies/encoders/rmvpe.pt`
3. So-VITS 主模型底模：
   - `model_assets/dependencies/base_models/44k/G_0.pth`
   - `model_assets/dependencies/base_models/44k/D_0.pth`
4. 扩散底模：
   - `model_assets/dependencies/base_models/44k/diffusion/model_0.pt`
5. NSF-HIFIGAN 声码器：
   - `model_assets/dependencies/vocoders/nsf_hifigan/`

建议顺序：

1. 在“选择依赖”里看第一个缺失项。
2. 点击“自动获取当前依赖”。
3. 如果自动获取失败，使用页面给出的下载链接手动下载。
4. 上传或选择本地文件导入。
5. 点击“刷新依赖状态”。
6. 重复直到训练前依赖与底模全部就绪。

## 5. 建立模型工作区

训练页里的“模型工作区”决定训练产物写到哪里。

规则：

1. 一个说话人（需要训练的语音角色）建一个模型工作区。
2. 模型名建议只用英文、数字、下划线或短横线。  

后续以 paimeng 为例，在 新建模型名 输入 paimeng ，点击 新建训练模型 即可。

新建后会生成如下文件：

```text
model_assets/workspaces/paimeng/
```

训练第二个说话人时，在 新建模型名 输入 "新的名字" ，点击 新建训练模型 即可。


## 6. 导入训练语音数据

在训练页的“训练语音数据”区域导入 wav。

数据要求：

1. 单说话人。
2. 尽量干声，无伴奏、低噪声、少混响。
3. 语音不要太长，限制10s内，显卡12g的话最多15s，再长的话容易在训练过程中爆内存。
4. 文件名不要包含奇怪符号。
5. 语音格式后缀仅支持 .wav .WAV。

导入后的数据位置

```text
model_assets/workspaces/paimeng/training_data/source/
```

## 7. 按页面步骤训练

训练页里按顺序执行：

1. `1. 重采样到工作区训练目录`
2. `2. 生成配置与文件列表`
3. `3. 提取特征`
4. `4. 启动主模型训练`
5. `5. 启动扩散训练`
6. `6. 训练音色增强索引`

也可以使用：

1. “一键执行 1-3 步”
2. “一键执行到主模型训练”

第一次验证建议先用少量数据确认：

1. 重采样能完成。
2. 配置生成能完成。
3. 特征提取能完成。
4. 主模型训练能正常启动并产出 `G_*.pth`。

训练页会显示当前任务状态、最近日志、错误/警告摘要。如果按钮点击后页面提示“正在启动中”，稍等即可；训练监控和推理页会在服务真正就绪后再尝试打开浏览器。

## 8. 打开训练监控

在训练页点击“打开训练监控”。

如果 TensorBoard 启动较慢，页面会先显示“训练监控正在启动中”，后台等端口就绪后自动打开浏览器。

也可以手动访问：

```text
http://127.0.0.1:6006
```

如果没有曲线，先确认是否已经启动过训练任务，并且 `model_assets/workspaces/` 下已经有训练日志。

## 9. 进入推理界面

训练页里可以点击“进入推理界面”。如果推理页启动较慢，页面会先显示“推理页面正在启动中”，后台等服务就绪后自动打开。

也可以单独启动：

```powershell
python -m src.app_infer
```

或者双击：

```text
launchers/启动推理界面.bat
```

推理页基本流程：

1. 在模型来源里选择训练工作区或已导入模型。
2. 选择主模型 `G_*.pth`。
3. 如有扩散模型、音色增强文件，可一起选择。
4. 点击“加载模型”。
5. 选择质量模式，推荐先用默认或极致质量。
6. 上传输入音频。
7. 点击“开始转换”。
8. 在右侧试听，最终文件会写入 `inference_data/outputs/`。

## 10. 首次验证建议

建议先用小闭环验证，不要一上来训练很久。

1. 准备 15-30 秒干声测试音频。
2. 训练页完成 1-3 步。
3. 主模型训练跑到能产出一个 `G_*.pth`。
4. 打开推理页加载这个模型。
5. 转换一段短音频。
6. 检查输出是否能正常播放。

听感重点：

1. 音准。
2. 咬字。
3. 音色相似度。
4. 断句连接。
5. 尾音稳定性。
6. 底噪和电音感。

## 11. 当前仓库已配套的改进

1. 训练页已负责依赖导入、底模检查、模型工作区、数据导入和 1-6 步训练流程。
2. ContentVec HF 依赖已细化到 `config.json` / `model.safetensors` 文件级状态。
3. “打开训练监控”和“进入推理界面”已补齐慢启动等待与可见提示。
4. 推理页支持训练工作区模型、已导入模型、质量模式和运行摘要。
5. 高质量默认值已统一收敛到 `src/quality_presets.py`。

## 12. 排查入口

如果遇到安装失败、CUDA 不可用、依赖缺失、训练中断、推理加载失败、音质异常等问题，请看：

- [Windows 常见问题排查](02_Windows常见问题排查.md)
