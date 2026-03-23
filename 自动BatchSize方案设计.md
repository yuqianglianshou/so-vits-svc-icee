# 自动 Batch Size 方案设计

## 目标

在当前项目里增加一个“自动寻找可用 batch size”的能力，尽量逼近 `so-vits-svc-fork` 那种：

- 更少手工试错
- 更容易把显存吃满
- 在不同显卡上拿到更接近上限的训练吞吐

同时保持下面这几条不被破坏：

- 现有固定 `batch_size` 的训练配置继续可用
- 训练页现有工作流尽量少改
- 多卡 DDP 主训练链不因为自动探测变得更脆

## 当前现状

### 当前训练脚本

当前主模型训练脚本是：

- [train.py](/Volumes/D/MyGitHub/so-vits-svc-icee/train_pipeline/train.py)

现状是：

- 直接读取 `hps.train.batch_size`
- 然后把它作为 `DataLoader(batch_size=...)` 的固定值
- 当前 `batch_size` 语义是“**每卡 batch size**”

也就是说，多卡训练时：

- 进程数 = GPU 数量
- 每个进程各自用同一个 `batch_size`

### 当前训练配置

当前配置模板在：

- [config_template.json](/Volumes/D/MyGitHub/so-vits-svc-icee/configs_template/config_template.json)
- [config_tiny_template.json](/Volumes/D/MyGitHub/so-vits-svc-icee/configs_template/config_tiny_template.json)

`train.batch_size` 现在是纯整数，没有自动模式。

### 当前训练页

训练页第 2 步会生成工作区配置：

- `logs/<模型名>/config.json`

主模型训练入口在：

- [tasks.py](/Volumes/D/MyGitHub/so-vits-svc-icee/train_ui/tasks.py)

目前只会直接启动：

```bash
python -m train_pipeline.train -c logs/<模型名>/config.json
```

## 为什么这个点值得做

当前项目里，最可能直接影响训练吞吐的几个点里，`batch_size` 是收益最直接的一项：

1. 配太小，GPU 利用率上不去
2. 配太大，又会直接 OOM
3. 不同显卡、不同显存、不同模型大小，最优值差异很大

所以“自动探测一个可用上限附近的值”是很有价值的。

## 约束与风险

自动 batch size 这件事在当前项目里不能粗暴做，主要有这几个风险。

### 1. 当前训练是多卡 DDP

[train.py](/Volumes/D/MyGitHub/so-vits-svc-icee/train_pipeline/train.py) 当前逻辑是：

- 先 `assert torch.cuda.is_available()`
- 再 `mp.spawn(run, nprocs=n_gpus, ...)`

如果我们直接在真实 DDP 训练过程中“边训练边试 batch size”，会把下面几件事耦在一起：

- OOM
- 分布式进程组初始化
- DataLoader 构建
- 检查点恢复

这会很难收。

### 2. OOM 不只是 DataLoader 问题

真正决定能不能跑下来的，不只是：

- `batch_size`

还包括：

- `segment_size`
- 模型大小
- 是否启用自动 F0
- 训练时显存碎片
- 当前 GPU 型号

所以自动 batch size 只能解决“当前配置下，找一个大致可跑的 batch size”，不能保证所有场景都绝对最优。

### 3. 探测阶段要能安全恢复

如果探测时发生 OOM，需要确保：

- 及时 `torch.cuda.empty_cache()`
- 不把 DDP 初始化搞残
- 不污染正式训练状态

## 推荐方案

我建议按 **两阶段** 做，而不是一步到位把探测逻辑塞进主训练循环。

### 阶段 A：先做独立探测脚本

新增一个独立脚本，例如：

- `train_pipeline/autobatch_probe.py`

作用：

- 只在单卡、单进程上做 batch size 探测
- 不进入正式 DDP 训练
- 只返回一个“建议 batch size”

#### 探测输入

- `--config logs/<模型名>/config.json`
- 可选：
  - `--device cuda:0`
  - `--start-batch-size 4`
  - `--max-batch-size 64`
  - `--max-trials 6`

#### 探测输出

输出一份结果，例如：

```json
{
  "status": "ok",
  "recommended_batch_size": 12,
  "tested_device": "cuda:0",
  "trials": [
    {"batch_size": 8, "ok": true},
    {"batch_size": 12, "ok": true},
    {"batch_size": 16, "ok": false, "reason": "oom"}
  ]
}
```

#### 探测方式

建议采用：

1. 只加载一个小批次
2. 构建一次 `net_g` / `net_d`
3. 跑一轮最小前向 + 反向
4. 捕获 `CUDA out of memory`
5. 用二分或指数上探方式寻找上限

这个阶段不写 checkpoint，不进入完整 epoch。

### 阶段 B：训练页与训练脚本接入“自动模式”

等探测脚本稳定后，再把自动模式接进工作流。

#### 配置语法建议

建议支持下面两种之一：

##### 方案 1：字符串模式

```json
"batch_size": "auto"
```

或：

```json
"batch_size": "auto-8-6"
```

含义：

- 初始 batch size = 8
- 最多探测 6 次

##### 方案 2：显式字段

```json
"batch_size": 8,
"auto_batch_size": true,
"auto_batch_start": 8,
"auto_batch_max_trials": 6
```

我更推荐 **方案 2**，原因是：

- 兼容当前 `batch_size` 是整数的设计
- 配置解析更简单
- 文档也更清楚

#### 正式训练接法

正式训练不建议在 `train.py` 里直接做复杂探测。更稳的做法是：

1. 训练页点击“启动主模型训练”
2. 如果发现：
   - `auto_batch_size=true`
3. 先运行：
   - `python -m train_pipeline.autobatch_probe ...`
4. 得到推荐值后：
   - 生成一个临时训练配置
   - 或通过环境变量传递解析后的整数 batch size
5. 再启动真正的：
   - `python -m train_pipeline.train ...`

也就是：

- **探测和正式训练分离**

这样风险最小。

## 为什么不建议第一版直接写进 `train.py`

因为当前 [train.py](/Volumes/D/MyGitHub/so-vits-svc-icee/train_pipeline/train.py) 是：

- DDP
- 手写训练循环
- 检查点恢复
- 多进程 spawn

如果把自动探测直接混进去，第一版最容易出现的问题是：

1. OOM 后进程状态不干净
2. 多卡环境下每张卡行为不一致
3. 探测逻辑和正式训练逻辑耦在一起，不好排错

## 与当前 DataLoader 调优项的关系

我们刚新增的 DataLoader 配置项在：

- `num_workers`
- `pin_memory`
- `persistent_workers`
- `prefetch_factor`

它们和自动 batch size 是互补关系：

- DataLoader 调优：减少“喂数瓶颈”
- 自动 batch size：提高“单步吞吐上限”

推荐顺序是：

1. 先让 DataLoader 配置可调
2. 再做自动 batch size 探测

这样探测出来的结果更有参考价值。

## 建议的第一版范围

第一版只做这些：

1. 新增 `train_pipeline/autobatch_probe.py`
2. 支持单卡探测
3. 支持生成推荐 batch size
4. 不改训练页 UI
5. 不改现有 `batch_size` 语义

也就是说，第一版目标只是：

- **让我们有一把“测当前配置建议 batch size”的尺子**

先不追求一步到位的全自动训练。

## 验收标准

如果后面真正开始实现，我建议验收标准按这几条来：

1. 固定 `batch_size` 的老配置完全不受影响
2. 探测脚本在单卡 GPU 环境下能返回推荐值
3. OOM 时不会把后续正式训练搞坏
4. 不同显存机器上，推荐值能明显区别开
5. 正式训练吞吐相比保守手填值有可观察提升

## 当前建议

就当前项目阶段而言，最稳的下一步不是直接改主训练入口，而是：

1. 先实现独立探测脚本
2. 在一台有 GPU 的机器上验证
3. 再决定要不要把它接进训练页按钮工作流

---

一句话总结：

**自动 batch size 值得做，但应该先做成“独立探测器”，不要第一刀就直接嵌进 DDP 主训练链。**
