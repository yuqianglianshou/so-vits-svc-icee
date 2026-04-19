# PC-NSF-HiFiGAN 接入评估

这份文档只回答一个问题：如果后续要把 `PC-NSF-HiFiGAN` 接进当前项目，应该怎么做、风险在哪里。

## 1. 总体结论

不要直接替换当前 `nsf_hifigan`。

推荐策略：

1. 新增并行路线：`pc-nsf-hifigan`。
2. 默认仍保持当前 `nsf-hifigan`。
3. 先做独立加载和推理验证。
4. 再接推理增强器。
5. 最后才评估扩散训练 / 扩散推理链。

原因：当前 `nsf_hifigan` 同时影响训练配置、扩散训练、扩散推理、增强器和训练页依赖管理。直接替换破坏面太大。

## 2. 低风险改动

### 2.1 资源注册

可能涉及：

- `src/path_utils.py`
- `src/app_train.py`
- `src/train_ui/pretrain.py`

建议：

1. 新增目录：`model_assets/dependencies/vocoders/pc_nsf_hifigan/`
2. 新增训练页资源项：`pc_nsf_hifigan`
3. 先标为实验依赖，不影响旧版 `nsf_hifigan`

### 2.2 配置枚举

可能涉及：

- `config_templates/config_template.json`
- `config_templates/diffusion_template.yaml`
- `src/train_pipeline/preprocess_flist_config.py`

建议允许配置值扩展为：

```text
nsf-hifigan
pc-nsf-hifigan
```

默认值仍保持 `nsf-hifigan`。

## 3. 中风险改动

### 3.1 Vocoder 适配层

可能涉及：

- `src/diffusion/vocoder.py`
- `src/vdecoder/`
- `src/modules/enhancer.py`

关键是把新版声码器包装成与旧版接近的接口：

```text
sample_rate()
hop_size()
dimension()
extract(...)
forward / infer(...)
```

风险点：

1. mel 维度是否一致。
2. hop size 是否一致。
3. sample rate 是否一致。
4. F0 输入形式是否一致。
5. 配置字段是否能被统一解析。

### 3.2 推理增强器接线

可能涉及：

- `src/modules/enhancer.py`
- `src/inference/infer_tool.py`

建议：

1. 先允许手动选择 `pc-nsf-hifigan`。
2. 不要默认打开。
3. 和旧版 `nsf-hifigan` 做 A/B 对比。

## 4. 高风险改动

### 4.1 扩散链

可能涉及：

- `src/train_pipeline/train_diff.py`
- `src/diffusion/unit2mel.py`
- `src/diffusion/solver.py`
- `src/diffusion/infer_gt_mel.py`

风险很高，因为扩散模型依赖声码器的 mel 契约。如果新版声码器的 mel 标准不同，旧扩散模型可能不能直接复用。

建议：

1. 不要让旧扩散模型直接走新版声码器。
2. 新版声码器先单独实验。
3. 真要接扩散，应训练新扩散模型做验证。

### 4.2 旧模型兼容性

旧模型工作区里通常写着：

```text
vocoder_name: nsf-hifigan
```

如果粗暴改默认值，旧模型可能出现：

1. 配置读不进。
2. mel 契约不一致。
3. 推理结果音质偏移。
4. 扩散模型不可用。

## 5. 推荐推进顺序

### 第一阶段：资源与加载验证

1. 新增资源目录。
2. 新增训练页实验依赖项。
3. 手动下载并验证文件结构。
4. 单独写最小加载测试。

### 第二阶段：独立推理验证

1. 新增 `vdecoder/pc_nsf_hifigan/`。
2. 在 `src/diffusion/vocoder.py` 注册新类型。
3. 用同一输入做旧版 / 新版 A/B。

### 第三阶段：增强器链验证

1. 扩展 `src/modules/enhancer.py`。
2. 推理页只作为实验选项展示。
3. 不影响默认推理路径。

### 第四阶段：扩散链验证

1. 确认 mel 契约。
2. 训练新版扩散模型。
3. 和旧扩散模型分开命名、分开配置。

## 6. 不建议做的事

1. 不要把新版资源覆盖到 `nsf_hifigan/`。
2. 不要直接改默认 `vocoder_name`。
3. 不要假设旧扩散模型能直接复用。
4. 不要在没有 A/B 验证前写进推荐配置。

## 7. 一句话总结

`PC-NSF-HiFiGAN` 可以作为实验路线接入，但它不是“替换一个模型文件”。正确做法是新增并行声码器路线，先验证推理，再评估扩散链，最后才考虑是否进入主线。
