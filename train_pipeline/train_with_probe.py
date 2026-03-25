"""主模型训练包装器：先自动探测 batch size，再按安全余量切入正式训练。"""

import argparse
import json
import os
import sys

from train_pipeline.autobatch_probe import (
    apply_safety_margin,
    generate_candidate_batch_sizes,
    run_single_probe,
)
import utils


def log_autobatch(message: str):
    """确保自动 batch size 日志在重定向到文件时也能及时落盘。"""
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="训练配置文件")
    parser.add_argument("-m", "--model", type=str, required=True, help="模型工作区名称")
    parser.add_argument("--device", type=str, default="cuda:0", help="用于探测的设备")
    parser.add_argument("--start-batch-size", type=int, default=None, help="起始每卡 batch size")
    parser.add_argument("--max-batch-size", type=int, default=64, help="探测上限")
    parser.add_argument("--max-trials", type=int, default=6, help="最多探测次数")
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=0.75,
        help="自动探测后的推荐安全系数；1.0 表示直接使用极限可用值。",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="只做 batch size 探测，不进入正式训练。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hps = utils.get_hparams_from_file(args.config)
    start_batch_size = args.start_batch_size or hps.train.batch_size
    candidate_batch_sizes = generate_candidate_batch_sizes(
        start_batch_size, args.max_batch_size, args.max_trials
    )

    log_autobatch(
        f"[AutoBatch] 已启用训练前自动探测，设备：{args.device}，"
        f"起始每卡 batch size：{start_batch_size}，安全系数：{args.safety_factor}"
    )
    trials = []
    last_success = None
    for batch_size in candidate_batch_sizes:
        result = run_single_probe(hps, batch_size, args.device)
        trials.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["ok"]:
            last_success = batch_size
            continue
        break

    if last_success is None:
        raise RuntimeError(
            "自动 batch size 探测未找到可用值，请先手动降低 batch_size，或关闭训练前自动探测。"
        )

    recommended_batch_size = apply_safety_margin(last_success, args.safety_factor)
    log_autobatch(
        f"[AutoBatch] 探测到的极限每卡 batch size：{last_success}；"
        f"按安全系数折算后的推荐值：{recommended_batch_size}。"
    )
    if args.probe_only:
        log_autobatch("[AutoBatch] 已完成独立探测，本次不会启动正式训练。")
        return

    log_autobatch(f"[AutoBatch] 即将按推荐值切入正式训练。")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "train_pipeline.train",
            "-c",
            args.config,
            "-m",
            args.model,
            "--batch-size",
            str(recommended_batch_size),
        ],
    )


if __name__ == "__main__":
    main()
