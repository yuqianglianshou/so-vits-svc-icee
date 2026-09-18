from __future__ import annotations

import argparse
from pathlib import Path

from src.infer_ui.bundle import export_infer_bundle


def main():
    parser = argparse.ArgumentParser(description="导出单文件推理包。")
    parser.add_argument("-m", "--model", required=True, help="主模型 G_*.pth 路径")
    parser.add_argument("-c", "--config", required=True, help="对应的 config.json 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 bundle 路径")
    args = parser.parse_args()

    output_path, bundle_meta = export_infer_bundle(
        Path(args.model).expanduser(),
        Path(args.config).expanduser(),
        Path(args.output).expanduser(),
    )
    extras = []
    if bundle_meta.get("has_diffusion"):
        extras.append("扩散")
    if bundle_meta.get("has_index"):
        extras.append("索引")
    if bundle_meta.get("has_cover"):
        extras.append("封面")
    if bundle_meta.get("has_description"):
        extras.append("说明")
    extras_text = f"；已包含：{', '.join(extras)}" if extras else "；未检测到额外资产"
    print(f"已导出单文件推理包：{output_path.as_posix()}{extras_text}")


if __name__ == "__main__":
    main()
