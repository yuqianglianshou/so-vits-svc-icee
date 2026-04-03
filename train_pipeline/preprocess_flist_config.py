import argparse
import json
import os
import re
import wave
from random import shuffle

from loguru import logger
from tqdm import tqdm

import diffusion.logger.utils as du
import utils

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')


def split_train_val(wavs: list[str]) -> tuple[list[str], list[str]]:
    """单说话人训练集划分。

    规则：
    1. 只要存在样本，就尽量保证 train 至少有 1 条；
    2. 验证集默认取少量样本，不再固定死取前 2 条；
    3. 小数据集场景下优先保证训练集非空。
    """
    total = len(wavs)
    if total == 0:
        return [], []
    if total == 1:
        return wavs[:], []

    # 默认给验证集留 10%，但至少 1 条；同时强制给训练集保留至少 1 条。
    val_count = max(1, total // 10)
    val_count = min(val_count, 2)
    val_count = min(val_count, total - 1)
    return wavs[val_count:], wavs[:val_count]

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    parser.add_argument("--config_out", type=str, default="configs/config.json", help="path to generated main config")
    parser.add_argument("--diff_config_out", type=str, default="configs/diffusion.yaml", help="path to generated diffusion config")
    parser.add_argument("--exp_dir", type=str, default="logs/44k/diffusion", help="path to diffusion experiment dir")
    parser.add_argument(
        "--speech_encoder",
        type=str,
        default="vec768l12",
        help="[内容编码器] 当前唯一主线为 vec768l12（底层已切到 transformers/HF）。",
    )
    # 极致音质配置：默认启用响度嵌入
    parser.add_argument("--vol_aug", action="store_true", help="[极致音质配置] 启用响度嵌入（默认已启用）")
    parser.add_argument("--no_vol_aug", action="store_true", help="[极致音质配置] 禁用响度嵌入")
    parser.add_argument("--tiny", action="store_true", help="Whether to train sovits tiny")
    args = parser.parse_args()
    
    # 极致音质配置：默认启用响度嵌入，除非明确禁用
    vol_aug = not args.no_vol_aug
    resolved_speech_encoder = utils.LEGACY_SPEECH_ENCODER_ALIASES.get(args.speech_encoder, args.speech_encoder)
    speech_encoder_spec = utils.get_speech_encoder_spec(resolved_speech_encoder)
    if resolved_speech_encoder not in utils.get_supported_speech_encoders():
        logger.warning(
            f"[内容编码器] 未识别的编码器 `{args.speech_encoder}`，已自动切换为 vec768l12。"
        )
        resolved_speech_encoder = "vec768l12"
        speech_encoder_spec = utils.get_speech_encoder_spec(resolved_speech_encoder)
    
    config_template =  json.load(open("configs_template/config_tiny_template.json")) if args.tiny else json.load(open("configs_template/config_template.json"))
    train = []
    val = []
    spk_dict = {}
    spk_id = 0
    speakers = sorted(
        [
            speaker
            for speaker in os.listdir(args.source_dir)
            if os.path.isdir(os.path.join(args.source_dir, speaker)) and not speaker.startswith(".")
        ]
    )
    if len(speakers) != 1:
        raise RuntimeError(
            f"单说话人模式要求 {args.source_dir} 下必须且只能有 1 个说话人目录，当前检测到 {len(speakers)} 个：{speakers}"
        )

    for speaker in tqdm(speakers):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = []

        for file_name in os.listdir(os.path.join(args.source_dir, speaker)):
            if not file_name.lower().endswith(".wav"):
                continue
            if file_name.startswith("."):
                continue

            file_path = "/".join([args.source_dir, speaker, file_name])

            if not pattern.match(file_name):
                logger.warning("Detected non-ASCII file name: " + file_path)

            if get_wav_duration(file_path) < 0.3:
                logger.info("Skip too short audio: " + file_path)
                continue

            wavs.append(file_path)

        shuffle(wavs)
        speaker_train, speaker_val = split_train_val(wavs)
        train += speaker_train
        val += speaker_val

    shuffle(train)
    shuffle(val)

    logger.info("Writing " + args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")

    logger.info("Writing " + args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")


    os.makedirs(os.path.dirname(args.train_list), exist_ok=True)
    os.makedirs(os.path.dirname(args.val_list), exist_ok=True)
    os.makedirs(os.path.dirname(args.config_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.diff_config_out), exist_ok=True)
    os.makedirs(args.exp_dir, exist_ok=True)

    d_config_template = du.load_config("configs_template/diffusion_template.yaml")
    d_config_template["model"]["n_spk"] = spk_id
    d_config_template["data"]["encoder"] = resolved_speech_encoder
    d_config_template["data"]["training_files"] = args.train_list
    d_config_template["data"]["validation_files"] = args.val_list
    d_config_template["env"]["expdir"] = args.exp_dir
    d_config_template["spk"] = spk_dict
    
    config_template["spk"] = spk_dict
    config_template["model"]["n_speakers"] = spk_id
    config_template["data"]["training_files"] = args.train_list
    config_template["data"]["validation_files"] = args.val_list
    config_template["model"]["speech_encoder"] = resolved_speech_encoder
    
    # 目前接入的 ContentVec 双路线都保持 768 维输出，这里统一由编码器元信息驱动。
    encoder_dim = speech_encoder_spec["ssl_dim"]
    config_template["model"]["ssl_dim"] = encoder_dim
    config_template["model"]["gin_channels"] = encoder_dim
    config_template["model"]["filter_channels"] = encoder_dim
    d_config_template["data"]["encoder_out_channels"] = speech_encoder_spec["diffusion_out_channels"]
    
    # 极致音质配置：默认启用响度嵌入
    config_template["train"]["vol_aug"] = config_template["model"]["vol_embedding"] = vol_aug

    if args.tiny:
        config_template["model"]["filter_channels"] = 512

    logger.info("Writing to " + args.config_out)
    with open(args.config_out, "w") as f:
        json.dump(config_template, f, indent=2)
    logger.info("Writing to " + args.diff_config_out)
    du.save_config(args.diff_config_out, d_config_template)
