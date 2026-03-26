# PC-NSF-HiFiGAN 接入改造清单

本文只回答一个工程问题：

- 如果后续要把 `PC-NSF-HiFiGAN` 接进当前项目，哪些模块需要改？
- 哪些是低风险、哪些是中风险、哪些是高风险？

当前项目的现实前提：

- 主模型配置默认使用 `nsf-hifigan`
- 扩散配置默认使用 `type: nsf-hifigan`
- 本地代码实现围绕旧版 `vdecoder/nsf_hifigan/` 写死
- 训练页的依赖检查、自动获取、说明文案，也都以旧版 `nsf_hifigan` 文件结构为前提

所以，这不是“换一个模型文件”的问题，而是一条新的 vocoder 路线接入问题。

## 总体判断

如果要做，建议按下面的策略推进：

1. 先新增一条并行路线：`pc-nsf-hifigan`
2. 不要直接覆盖现有 `nsf-hifigan`
3. 先只打通“独立加载 + 推理验证”
4. 再接“扩散链”
5. 最后才考虑是否替换当前默认值

原因很简单：当前 `nsf-hifigan` 已经同时接在

- 主模型配置
- 扩散训练
- 扩散推理
- 增强器
- 训练页依赖管理

这意味着一旦直接替换，破坏面会很大。

---

## 低风险

这些模块最适合先动，因为它们主要是“增加并行入口”，不会立刻打断当前主线。

### 1. 路径与资源注册层

相关文件：
- [app_train.py](/Volumes/D/MyGitHub/so-vits-svc-icee/app_train.py)
- [path_utils.py](/Volumes/D/MyGitHub/so-vits-svc-icee/path_utils.py)

当前现状：
- 训练页只认识 `pretrain/vocoders/nsf_hifigan/`
- 自动获取、缺失检查、说明文案也都只写旧包名

建议改法：
- 新增一个并行资源键，例如：
  - `pc_nsf_hifigan`
- 新增对应目录，例如：
  - `pretrain/vocoders/pc_nsf_hifigan/`
- 新增获取规则、缺失检查规则，但先不要替换旧版默认值

为什么是低风险：
- 这层只是让项目“认识”新资源
- 不会直接改变推理和训练行为

### 2. 配置枚举与模板字段扩展

相关文件：
- [configs_template/config_template.json](/Volumes/D/MyGitHub/so-vits-svc-icee/configs_template/config_template.json)
- [configs_template/diffusion_template.yaml](/Volumes/D/MyGitHub/so-vits-svc-icee/configs_template/diffusion_template.yaml)
- [train_pipeline/preprocess_flist_config.py](/Volumes/D/MyGitHub/so-vits-svc-icee/train_pipeline/preprocess_flist_config.py)

当前现状：
- `vocoder_name` 固定是 `nsf-hifigan`
- 扩散模板里的 `vocoder.type` 也是 `nsf-hifigan`

建议改法：
- 允许配置值扩展成：
  - `nsf-hifigan`
  - `pc-nsf-hifigan`
- 但默认值仍保持旧版

为什么是低风险：
- 只是先把配置通道留出来
- 不会强制现有项目切线

### 3. 文档与训练页提示

相关文件：
- [README.md](/Volumes/D/MyGitHub/so-vits-svc-icee/README.md)
- [项目分析报告.md](/Volumes/D/MyGitHub/so-vits-svc-icee/项目分析报告.md)
- [app_train.py](/Volumes/D/MyGitHub/so-vits-svc-icee/app_train.py)

建议改法：
- 明确区分：
  - 当前稳定主线：`nsf-hifigan`
  - 实验路线：`pc-nsf-hifigan`
- 不要在尚未打通前把新版写成推荐主线

为什么是低风险：
- 只是降低后续误用风险

---

## 中风险

这些模块开始真正碰到“模型怎么被加载、怎么被调用”的问题。它们通常可以做，但要小心输入输出契约是否一致。

### 4. 新版 vocoder 加载适配层

相关文件：
- [diffusion/vocoder.py](/Volumes/D/MyGitHub/so-vits-svc-icee/diffusion/vocoder.py)
- [vdecoder/nsf_hifigan/models.py](/Volumes/D/MyGitHub/so-vits-svc-icee/vdecoder/nsf_hifigan/models.py)
- [modules/enhancer.py](/Volumes/D/MyGitHub/so-vits-svc-icee/modules/enhancer.py)

当前现状：
- `diffusion/vocoder.py` 只认识：
  - `nsf-hifigan`
  - `nsf-hifigan-log10`
- 背后写死依赖旧版 `vdecoder.nsf_hifigan` 的：
  - `load_config`
  - `load_model`
  - `STFT`

建议改法：
- 新建一条并行实现，例如：
  - `vdecoder/pc_nsf_hifigan/`
- 在 `diffusion/vocoder.py` 中新增：
  - `pc-nsf-hifigan`
- 让它能像旧版一样暴露统一接口：
  - `sample_rate()`
  - `hop_size()`
  - `dimension()`
  - `extract(...)`
  - `forward/infer(...)`

为什么是中风险：
- 如果新版接口能被包装成与旧版相同，这层风险可控
- 但如果 mel 规范、F0 输入形式、配置解析方式不同，就需要写比较厚的适配层

### 5. 推理增强器接线

相关文件：
- [modules/enhancer.py](/Volumes/D/MyGitHub/so-vits-svc-icee/modules/enhancer.py)
- [inference/infer_tool.py](/Volumes/D/MyGitHub/so-vits-svc-icee/inference/infer_tool.py)

当前现状：
- 增强器直接写死：
  - `Enhancer('nsf-hifigan', ...)`
- `modules/enhancer.py` 也直接 import 旧版实现

建议改法：
- 扩展增强器类型：
  - `nsf-hifigan`
  - `pc-nsf-hifigan`
- 先保证新版至少能被单独加载和推理，不要一开始就默认替换旧版

为什么是中风险：
- 增强器链通常比主训练短，适合先做实验验证
- 但如果新版 vocoder 的 `extract/infer` 契约和旧版差很多，这层也会跟着变复杂

### 6. 训练页自动获取与依赖检查

相关文件：
- [app_train.py](/Volumes/D/MyGitHub/so-vits-svc-icee/app_train.py)

当前现状：
- 自动获取的声码器包就是旧版：
  - `nsf_hifigan_20221211.zip`
- 缺失检查也按旧目录结构判断

建议改法：
- 增加新版资源项，但先独立显示为实验依赖
- 不要复用旧资源键，避免覆盖现有稳定资源

为什么是中风险：
- 这层逻辑简单
- 但一旦资源包结构不同，解压、嵌套目录整理、有效性校验都要改

---

## 高风险

这些模块一旦动，就不再只是“加一条新 vocoder 路线”，而是开始影响训练产物兼容性、扩散链假设、现有模型可用性。

### 7. 扩散训练与扩散推理主链

相关文件：
- [train_pipeline/train_diff.py](/Volumes/D/MyGitHub/so-vits-svc-icee/train_pipeline/train_diff.py)
- [diffusion/unit2mel.py](/Volumes/D/MyGitHub/so-vits-svc-icee/diffusion/unit2mel.py)
- [diffusion/solver.py](/Volumes/D/MyGitHub/so-vits-svc-icee/diffusion/solver.py)
- [diffusion/infer_gt_mel.py](/Volumes/D/MyGitHub/so-vits-svc-icee/diffusion/infer_gt_mel.py)

当前现状：
- 扩散训练/推理都默认通过 `Vocoder(...)` 拿 mel extractor 和波形还原器
- 它假设 vocoder 的：
  - mel 维度
  - hop size
  - sample rate
  - f0 兼容性
  与旧版一致

为什么高风险：
- 这条线一旦 mel 抽取标准不一致，扩散模型本身的训练目标就会改变
- 不是简单“能加载就行”，而是会影响旧扩散模型是否还能用
- 也会影响新扩散模型和旧主模型之间的搭配

建议：
- 这一层必须在“独立推理验证通过”之后再动
- 更稳的做法是把新版 vocoder 先仅用于新实验，不碰旧扩散模型

### 8. 现有已训练模型兼容性

相关文件：
- [inference/infer_tool.py](/Volumes/D/MyGitHub/so-vits-svc-icee/inference/infer_tool.py)
- [logs/<模型名>/config.json](/Volumes/D/MyGitHub/so-vits-svc-icee/logs/阜宁哪/config.json)
- [logs/<模型名>/diffusion.yaml](/Volumes/D/MyGitHub/so-vits-svc-icee/logs/阜宁哪/diffusion.yaml)

为什么高风险：
- 你现在已有模型工作区都写着：
  - `vocoder_name: nsf-hifigan`
  - `type: nsf-hifigan`
- 如果粗暴替换默认值，旧模型可能在推理时出现：
  - 读不进
  - mel 契约不一致
  - 音质明显偏移

建议：
- 不要直接替换默认值
- 先以新配置值新增一条实验路线
- 旧模型工作区继续保持旧 vocoder

### 9. 主模型训练主线是否切新版 vocoder

相关文件：
- [models.py](/Volumes/D/MyGitHub/so-vits-svc-icee/models.py)
- [configs_template/config_template.json](/Volumes/D/MyGitHub/so-vits-svc-icee/configs_template/config_template.json)

为什么高风险：
- 这一步本质上是在改“当前项目默认声码器假设”
- 会连带影响：
  - 新训练模型的产物风格
  - 旧底模兼容性
  - 文档、模板、训练页默认值
- 如果新版 vocoder 并不完全兼容旧训练目标，这一步会变成一轮真正的主线迁移

建议：
- 只有在下面三件事全部成立后，才考虑这一步：
  1. 新版 vocoder 已能独立推理
  2. 扩散链已验证可用
  3. 至少有一轮真实训练/推理 A/B 结果证明值得切

---

## 推荐推进顺序

### 第一阶段：最小可行验证

1. 新增本地目录与资源注册
2. 新增配置枚举 `pc-nsf-hifigan`
3. 新建并行 vocoder 适配层
4. 先做“独立加载 + 独立推理验证”

目标：
- 证明新版 vocoder 能在当前仓库里被正常加载和调用
- 不碰现有默认主线

### 第二阶段：接入实验推理链

5. 让增强器支持 `pc-nsf-hifigan`
6. 让推理链可按配置选择新版 vocoder
7. 用现有模型做有限 A/B 推理试听

目标：
- 先判断听感上有没有进一步研究价值

### 第三阶段：扩散链验证

8. 在扩散训练/推理里新增新版 vocoder 实验路线
9. 单独验证 mel 提取、sample rate、hop size、推理质量

目标：
- 证明它不是只能做独立推理，而是能进入我们当前扩散体系

### 第四阶段：是否考虑主线迁移

10. 只有在 A/B 结果明确后，才讨论是否替换当前默认的 `nsf-hifigan`

---

## 一句话总结

如果要把 `PC-NSF-HiFiGAN` 接进当前项目，真正的难点不在“下载新模型”，而在：

- 统一 vocoder 接口
- 保持旧模型兼容
- 处理扩散链对 mel/vocoder 契约的依赖

所以最稳的路线一定是：

- 先并行新增
- 再独立验证
- 最后才考虑是否替换当前默认主线
