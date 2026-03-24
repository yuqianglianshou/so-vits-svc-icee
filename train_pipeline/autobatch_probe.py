"""单卡自动 batch size 探测器。

第一版只做一件事：在当前配置下，用真实训练样本跑一轮最小训练步，
找到一个可用的每卡 batch size 建议值。

设计目标：
1. 不进入 DDP 主训练链；
2. 不写 checkpoint；
3. OOM 后可安全回收显存；
4. 输出结构化 JSON，便于后续接训练页或脚本。
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from torch.cuda.amp import autocast

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import MultiPeriodDiscriminator, SynthesizerTrn
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="待探测的训练配置文件")
    parser.add_argument("--device", type=str, default="cuda:0", help="用于探测的单卡设备")
    parser.add_argument(
        "--start-batch-size",
        type=int,
        default=None,
        help="起始每卡 batch size；默认使用配置里的 train.batch_size",
    )
    parser.add_argument("--max-batch-size", type=int, default=64, help="探测上限")
    parser.add_argument("--max-trials", type=int, default=6, help="最多探测次数")
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=0.75,
        help="推荐 batch size 的安全系数；1.0 表示直接使用极限可用值。",
    )
    return parser.parse_args()


def build_probe_batch(dataset, collate_fn, batch_size):
    """从真实数据集中取样，直接拼一个 batch，避免 DataLoader 干扰显存探测。"""
    if len(dataset) == 0:
        raise RuntimeError("训练集为空，无法进行 batch size 探测。")
    batch = [dataset[index % len(dataset)] for index in range(batch_size)]
    return collate_fn(batch)


def move_batch_to_device(items, device):
    c, f0, spec, y, spk, lengths, uv, volume = items
    return (
        c.to(device, non_blocking=True),
        f0.to(device, non_blocking=True),
        spec.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
        spk.to(device, non_blocking=True),
        lengths.to(device, non_blocking=True),
        uv.to(device, non_blocking=True),
        None if volume is None else volume.to(device, non_blocking=True),
    )


def release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_probe(hps, batch_size, device):
    """用指定 batch size 跑一轮最小训练步，返回是否成功。"""
    dataset = TextAudioSpeakerLoader(
        hps.data.training_files,
        hps,
        all_in_mem=hps.train.all_in_mem,
    )
    collate_fn = TextAudioCollate()

    half_type = torch.bfloat16 if hps.train.half_type == "bf16" else torch.float16

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)

    try:
        items = build_probe_batch(dataset, collate_fn, batch_size)
        c, f0, spec, y, spk, lengths, uv, volume = move_batch_to_device(items, device)
        g = spk

        net_g.zero_grad(set_to_none=True)
        net_d.zero_grad(set_to_none=True)

        with autocast(enabled=hps.train.fp16_run, dtype=half_type):
            y_hat, ids_slice, z_mask, (
                z,
                z_p,
                m_p,
                logs_p,
                m_q,
                logs_q,
            ), pred_lf0, norm_lf0, lf0 = net_g(
                c,
                f0,
                uv,
                spec,
                g=g,
                c_lengths=lengths,
                spec_lengths=lengths,
                vol=volume,
            )

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc.backward()

        net_g.zero_grad(set_to_none=True)

        with autocast(enabled=hps.train.fp16_run, dtype=half_type):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

        loss_mel = (
            torch.nn.functional.l1_loss(y_mel.float(), y_hat_mel.float()) * hps.train.c_mel
        )
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_lf0 = (
            torch.nn.functional.mse_loss(pred_lf0.float(), lf0.float())
            if hps.model.use_automatic_f0_prediction
            else 0
        )
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
        loss_gen_all.backward()

        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2)
        return {
            "ok": True,
            "batch_size": batch_size,
            "peak_memory_mb": peak_memory_mb,
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2)
            return {
                "ok": False,
                "batch_size": batch_size,
                "reason": "oom",
                "error": str(exc).splitlines()[0],
                "peak_memory_mb": peak_memory_mb,
            }
        raise
    finally:
        del net_g
        del net_d
        release_cuda_memory()


def generate_candidate_batch_sizes(start_batch_size, max_batch_size, max_trials):
    candidates = []
    current = start_batch_size
    for _ in range(max_trials):
        if current > max_batch_size:
            break
        candidates.append(current)
        current *= 2
    if not candidates:
        candidates.append(start_batch_size)
    return candidates


def apply_safety_margin(max_supported_batch_size, safety_factor):
    """把极限可用 batch size 转成更适合日常使用的推荐值。"""
    if max_supported_batch_size is None:
        return None
    recommended = int(max_supported_batch_size * safety_factor)
    if recommended < 1:
        return 1
    if recommended > max_supported_batch_size:
        return max_supported_batch_size
    return recommended


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("自动 batch size 探测需要可用 GPU，当前未检测到 CUDA。")

    device = torch.device(args.device)
    hps = utils.get_hparams_from_file(args.config)
    start_batch_size = args.start_batch_size or hps.train.batch_size
    candidate_batch_sizes = generate_candidate_batch_sizes(
        start_batch_size, args.max_batch_size, args.max_trials
    )

    trials = []
    last_success = None

    for batch_size in candidate_batch_sizes:
        result = run_single_probe(hps, batch_size, device)
        trials.append(result)
        if result["ok"]:
            last_success = batch_size
            continue
        break

    recommended_batch_size = apply_safety_margin(last_success, args.safety_factor)

    output = {
        "status": "ok" if last_success is not None else "failed",
        "config": str(Path(args.config)),
        "tested_device": str(device),
        "start_batch_size": start_batch_size,
        "max_batch_size": args.max_batch_size,
        "max_supported_batch_size": last_success,
        "safety_factor": args.safety_factor,
        "recommended_batch_size": recommended_batch_size,
        "trials": trials,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
