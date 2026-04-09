# Windows 继续开发上下文

这份文档用于在 Windows 机器上继续开发或训练时，快速恢复当前项目状态。

目标是减少“我上次改到哪里了”“现在主线是什么”“哪些地方已经改过”的反复确认成本。

---

## 1. 当前项目主线

### 1.1 环境基线

当前项目的稳定环境基线是：

- Python `3.11`
- 虚拟环境目录建议：
  - `.venv311`

Windows + NVIDIA 推荐安装顺序：

```powershell
python -m venv .venv311
.venv311\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

安装后建议验证：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

如果 `torch.cuda.is_available()` 为 `True`，说明 GPU 训练环境已打通。

---

## 2. 当前模型主线

### 2.1 内容编码器

当前项目已经切到：

- `vec768l12`

并且底层固定为：

- `Transformers / HF ContentVec`

本地固定目录是：

- `model_assets/dependencies/encoders/contentvec_hf/`

当前不再把 `fairseq ContentVec` 当作主线。

兼容旧配置时：

- `vec768l12_hf`
- `vec768l12_fairseq`

会自动归一到：

- `vec768l12`

### 2.2 F0 路线

当前主线固定使用：

- `rmvpe`

非 `rmvpe` 设置会自动切回 `rmvpe`。

### 2.3 推理/训练的核心理论骨架

当前项目主线可以压缩成：

1. `ContentVec` 提取内容
2. `RMVPE` 提取 F0
3. `SynthesizerTrn + vocoder` 完成目标歌声音色生成

如果想看更完整说明，请直接看：

- [项目理论基础与训练流程说明.md](项目理论基础与训练流程说明.md)

---

## 3. 当前训练页主流程

训练页入口：

- `app_train.py`

当前训练页主线已经整理为：

1. 重采样
2. 生成配置与文件列表
3. 提取特征
4. 主模型训练
5. 扩散训练
6. 训练索引

### 3.1 训练前依赖自动获取

当前训练页已支持自动获取这些依赖：

- `ContentVec HF`
- `RMVPE`
- `NSF-HIFIGAN`
- `G_0.pth`
- `D_0.pth`
- `model_0.pt`

也支持手动导入作为兜底。

### 3.2 数据准备主路径

当前主模型训练的数据组织已经优化过：

- 第 3 步会生成：
  - `*.train.pt`

这个统一特征包当前包含：

- `audio`
- `soft`
- `f0`
- `uv`
- `spec`
- `volume`

主模型训练现在优先且主要依赖：

- `*.train.pt`

不再把原始 `wav + 多个散文件` 当作主训练主路径。

### 3.3 训练页第 3 步完成判断

当前训练页状态面板里会显示：

- `trainpt=<数量>`

并且第 3 步要视为完成，需要：

- `soft / f0 / spec / trainpt`

数量都和 `wav` 对齐。

---

## 4. 当前已经完成的重要工程改造

### 4.1 结构整理

已经完成的分包：

- `train_ui/`
- `train_pipeline/`
- `tools/`
- `services/`
- `infer_ui/`

也就是说：

- 训练页 UI 逻辑已从 `app_train.py` 拆出大量辅助模块
- 推理页 UI 逻辑已从 `app_infer.py` 拆出大量辅助模块

### 4.2 训练吞吐优化第一阶段

已经做过：

1. 自动 batch size 第一版
2. DataLoader 调优入口
3. 主模型训练统一特征包 `*.train.pt`

对应文档：

- [自动BatchSize方案设计.md](自动BatchSize方案设计.md)
- [数据文件组织重构分析.md](数据文件组织重构分析.md)
- [项目分析报告.md](项目分析报告.md)

### 4.3 内置峰值归一化已删除

当前项目已明确假设：

- **训练音频应由用户自己提前处理好**

因此项目内已经删除：

- 旧的峰值归一化逻辑
- `--skip_loudnorm`

当前 `resample.py` 只负责：

- 基础静音裁剪
- 重采样

不再对音量、电平、响度做内置处理。

---

## 5. 训练是否可以中断并续训

### 5.1 主模型训练

可以中断，并且通常可以接着上次训练继续。

主模型 checkpoint 主要在：

- `model_assets/workspaces/<模型名>/G_*.pth`
- `model_assets/workspaces/<模型名>/D_*.pth`

只要满足下面几个条件，再次启动训练时通常就会沿着已有 checkpoint 继续：

1. 还是同一个模型工作区
2. `model_assets/workspaces/<模型名>/config.json` 没改成完全不同结构
3. 旧 checkpoint 没被删掉

### 5.2 扩散训练

扩散训练也有自己的 checkpoint 链，但它和主模型训练是两套东西。

同样要保证：

1. 还是同一个模型目录
2. 扩散配置没被结构性改坏
3. 扩散 checkpoint 还在

### 5.3 什么时候不适合续训

这些情况要谨慎：

- 改了模型结构参数
- 改了 encoder 维度
- 改了 speaker 结构
- 换了模型目录
- 删了旧 checkpoint

---

## 6. 当前 `requirements.txt` 的状态

已经做过一轮精简：

### 已移除

- `tensorboardX`

原因：

- 当前训练监控实际用的是 `torch.utils.tensorboard.SummaryWriter`

### 已改为可选安装

- `onnx`
- `onnxoptimizer`
- `onnxsim`

因为这几项主要用于：

- ONNX 导出 / 优化

不再作为默认必装依赖。

### 已从顶层显式清单移除

- `fastapi`
- `pydantic`
- `starlette`

原因：

- 当前代码没有直接 import 它们
- 它们作为 `gradio` 的传递依赖由 `gradio` 自己解析

### 当前仍保留且直接使用

- `huggingface_hub`

原因：

- `app_train.py` 里直接调用了 `hf_hub_download(...)`

### 当前仍偏老但暂不建议随便升级

- `numpy`
- `scipy`
- `librosa`
- `numba`
- `llvmlite`

原因：

- 它们属于当前稳定音频栈
- 不是当前最高收益改动
- 如果要动，应单独作为“音频栈升级”来做

---

## 7. Windows 机器上继续开发时最值得先看的文件

如果你回到 Windows 机器，最先看这些：

- [项目分析报告.md](项目分析报告.md)
- [项目理论基础与训练流程说明.md](项目理论基础与训练流程说明.md)
- [最佳配置推荐.md](最佳配置推荐.md)
- [Windows_GPU小白操作与验证指南.md](Windows_GPU小白操作与验证指南.md)
- [自动BatchSize方案设计.md](自动BatchSize方案设计.md)
- [数据文件组织重构分析.md](数据文件组织重构分析.md)

---

## 8. 当前最值得继续推进的方向

如果 Windows 机器已经可以开始训练，那么下一阶段最值得做的是：

1. 在真机上验证自动 batch size 的推荐结果是否合理
2. 观察 DataLoader 调优项对 GPU 吞吐的影响
3. 用新的 `*.train.pt` 主路径做一轮真实训练，确认收益
4. 再决定是否继续做更深的数据组织优化

不建议现在优先做的：

1. 大改 UI 框架
2. 直接上 QuickVC / MS-iSTFT
3. 把旧音频栈一次性升级到最新

---

## 9. 一句话总结

当前 Windows 机器上继续开发/训练时，可以把项目理解成：

- 环境基线已经稳定到 Python 3.11
- 内容编码器主线已经切到 HF ContentVec
- 训练页和推理页已经做过结构整理
- 自动依赖获取已经打通
- 训练吞吐优化第一阶段已经落地
- 现在最值得做的是**真机训练验证**，而不是继续大改架构

