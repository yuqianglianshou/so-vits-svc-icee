# Windows GPU 小白操作与验证指南

## 提交前自检

如果你在 Mac 上先改了页面、流程或脚本，再准备同步到 Windows + GPU 机器验证，建议先在项目目录执行：

```bash
python3 scripts/verify_stability_fixes.py
python3 scripts/verify_app_smoke.py
```

这两条不会替代 Windows 真机验证，但可以先排除最近入口、训练流程和推理流程有没有明显回退。

这份指南用于你在 Windows + NVIDIA GPU 机器上验证和使用本项目，目标是尽量少命令、可视化操作优先。

## 1. 先确认硬件与软件

1. 系统：Windows 10/11 64 位  
2. 显卡：NVIDIA（建议显存 >= 8GB）  
3. 驱动：安装最新 NVIDIA 驱动  
4. Python：建议 3.11  
5. Git：已安装并可用

## 2. 下载项目并安装依赖

说明：

1. 当前项目统一使用 Python 自带的 `venv`
2. 虚拟环境固定创建在项目根目录：`.venv311`
3. 不再使用 Conda / Anaconda / 其他第三方环境管理器作为默认方案

在 PowerShell 中执行：

```powershell
git clone <你的仓库地址>
cd so-vits-svc-icee
python -m venv .venv311
.venv311\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

说明：

1. 现在仓库只保留一个 `requirements.txt`
2. Windows 会自动安装 `onnxruntime-gpu`
3. macOS 会自动跳过不适合本地 WebUI 的 `onnxsim`，并固定兼容版本的 `setuptools`
4. 不再需要手动区分 `requirements_win.txt`、`requirements_mac_ui.txt`、`requirements_onnx_encoder.txt`
5. 后续所有 `.bat` 启动脚本都默认使用项目根目录下的 `.venv311`
6. 如果 `torch.cuda.is_available()` 为 `True`，说明 GPU 训练环境已经打通

安装完成后，建议立刻验证一次 GPU 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

如果输出中的 `torch.cuda.is_available()` 为 `True`，说明当前这台 Windows + NVIDIA 机器已经可以进入训练阶段。

## 3. 准备必要模型文件

必须准备：

1. 语音编码器权重，放到 `model_assets/dependencies/encoders/`
2. 训练底模，放到 `model_assets/dependencies/base_models/44k/`：
   - `model_assets/dependencies/base_models/44k/G_0.pth`
   - `model_assets/dependencies/base_models/44k/D_0.pth`
   - `model_assets/dependencies/base_models/44k/diffusion/model_0.pt`
2. 你要推理的模型与配置，例如：
   - `model_assets/workspaces/44k/G_xxx.pth`
   - `model_assets/workspaces/44k/config.json`

可选（但建议，用于更好效果）：

1. 扩散模型：
   - `model_assets/workspaces/44k/diffusion/config.yaml`
2. 特征检索文件：
   - `model_assets/workspaces/44k/feature_and_index.pkl`
3. 声码器：
   - `model_assets/dependencies/vocoders/nsf_hifigan/`

## 4. 启动可视化界面（推荐）

推荐优先直接进入训练页面：

```powershell
启动训练界面.bat
```

或直接运行：

```powershell
python -m src.app_train
```

训练页面顶部会先检查训练前依赖与底模，然后继续数据准备和训练流程。

如果你只想单独打开歌声转换页面，也可以：

```powershell
python -m src.app_infer
```

打开浏览器后按以下顺序操作：

1. 在“模型文件”区域先补齐基础必选：
   - `So-VITS 模型 .pth`
   - `模型配置 .json`
2. 如果你追求最佳效果，再补齐高音质增强文件：
   - `音质增强模型 .pt`
   - `音质增强配置 .yaml`
   - `音色增强文件 .pkl` 或聚类文件
3. 观察“文件完整度”面板：
   - 绿色越多，越接近最佳质量
   - 缺少基础模型时不要继续
4. 点击“加载模型”
5. 在“高音质参数”区域：
   - 默认选择 `极致质量`
   - 一般只改 `变调`
   - 点击“生成运行摘要”可记录本次测试条件
6. 上传输入音频并点击“开始转换”
7. 在右侧试听，最终文件在 `inference_data/outputs/`

## 5. 首次验证建议（10 分钟版本）

1. 准备一段 15-30 秒干声（无伴奏、噪声低）
2. 保持默认 `极致质量`，只按需要修改 `变调`
3. 分别测试：
   - 仅 So-VITS（不加载扩散）
   - So-VITS + 音质增强
   - So-VITS + 音质增强 + 音色增强
4. 对比听感：音准、齿音、气声、尾音、断句

## 6. 训练模型（可选）

若你要训练自己的模型，按顺序执行：

```powershell
python -m src.train_pipeline.preprocess_flist_config
python -m src.train_pipeline.preprocess_hubert_f0
python -m src.train_pipeline.train -m 44k
```

扩散训练（可选）：

```powershell
python -m src.train_pipeline.train_diff -m 44k
```

特征检索（可选）：

```powershell
python -m src.train_pipeline.train_index
```

如果你更希望用界面方式管理训练流程，可以启动：

```powershell
python -m src.app_train
```

默认地址：

```text
http://127.0.0.1:7861
```

训练控制台第一版支持：

1. 数据集与关键文件状态扫描
2. 一键执行重采样、生成配置、特征预处理
3. 启动主模型训练、扩散训练、音色增强索引训练
4. 查看当前任务状态与最近日志
5. 打开 TensorBoard

## 7. 常见问题排查

1. 显存不足：
   - 降低 batch size
   - 缩短输入音频
   - 先关闭扩散验证主流程
2. 声音跑调：
   - 关闭自动 F0（唱歌场景）
   - 使用 rmvpe
3. 咬字差但音色像：
   - 降低检索比例（如从 0.5 降到 0.3）
4. 连接处不自然：
   - 保持 `pad_seconds=0.5`
   - 避免过度切片

## 8. 你这版仓库中已做的改进（和这份指南配套）

1. 修复了推理返回值解包不一致导致的 API 崩溃
2. 修复了当前模型音色名称解析问题
3. 修复了 DDP 训练采样器问题
4. 修复了预处理多进程对象传递风险
5. 界面已重构为“基础必选 + 高音质增强文件 + 单页转换流程”
6. 增加了“文件完整度”提示、质量模式切换与更明确的结果说明
7. 高质量默认值已统一收敛到 `quality_presets.py`，WebUI 和命令行推理使用同一套预设
