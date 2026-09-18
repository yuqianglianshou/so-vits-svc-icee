# 贡献指南

感谢参与改进。当前主线优先保证 Windows 单说话人训练与离线推理流程稳定。

## 开始前

1. 使用 Python 3.11。
2. 从 `dev_icee` 创建范围明确的分支。
3. 安装项目依赖和开发依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

4. Bug 修复先提供可复现步骤和失败测试。
5. 不要提交模型权重、训练音频、推理输出、日志或虚拟环境。

## 修改原则

- 保持改动小而可审查，不改无关文件。
- 不删除错误处理来隐藏问题。
- 新增或修改代码时提供必要的中文注释。
- 模型结构、训练参数、目录和文件格式变化必须单独设计并说明兼容策略。
- 用户可见行为变化必须同步更新 README 或 docs。

## 提交前检查

```bash
python -m ruff check src scripts tests
python -m compileall -q src scripts tests
python scripts/verify_app_smoke.py
python scripts/verify_stability_fixes.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
python scripts/check_release.py
```

完整发布流程见 [开发与发布检查](docs/90_开发与发布检查.md)。

## Pull Request

PR 应说明：

- 用户问题和解决结果。
- 修改文件和不在范围内的内容。
- 实际运行的验证命令和结果。
- 模型、配置、目录与旧版本兼容性。
- 用户可见变化的文档和截图。
