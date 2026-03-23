import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from random import shuffle

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

import diffusion.logger.utils as du
import utils
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = None
dconfig = None
sampling_rate = None
hop_length = None
speech_encoder = None


def load_runtime_configs(config_path: str, diff_config_path: str):
    global hps, dconfig, sampling_rate, hop_length, speech_encoder
    hps = utils.get_hparams_from_file(config_path)
    dconfig = du.load_config(diff_config_path)
    sampling_rate = hps.data.sampling_rate
    hop_length = hps.data.hop_length
    speech_encoder = hps["model"]["speech_encoder"]


def process_one(filename, hmodel, f0p, device, diff=False, mel_extractor=None):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    soft_path = filename + ".soft.pt"
    train_bundle_path = filename + ".train.pt"
    c = None
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)

    f0_path = filename + ".f0.npy"
    f0 = uv = None
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(f0p,sampling_rate=sampling_rate, hop_length=hop_length,device=None,threshold=0.05)
        f0,uv = f0_predictor.compute_f0_uv(
            wav
        )
        np.save(f0_path, np.asanyarray((f0,uv),dtype=object))


    spec_path = filename.replace(".wav", ".spec.pt")
    spec = None
    if not os.path.exists(spec_path):
        # Process spectrogram
        # The following code can't be replaced by torch.FloatTensor(wav)
        # because load_wav_to_torch return a tensor that need to be normalized

        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sr, hps.data.sampling_rate
                )
            )

        #audio_norm = audio / hps.data.max_wav_value

        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_path)

    volume = None
    if diff or hps.model.vol_embedding:
        volume_path = filename + ".vol.npy"
        volume_extractor = utils.Volume_Extractor(hop_length)
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            np.save(volume_path, volume.to('cpu').numpy())

    if diff:
        mel_path = filename + ".mel.npy"
        if not os.path.exists(mel_path) and mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            np.save(mel_path, mel)
        aug_mel_path = filename + ".aug_mel.npy"
        aug_vol_path = filename + ".aug_vol.npy"
        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        keyshift = random.uniform(-5, 5)
        if mel_extractor is not None:
            aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift = keyshift)
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
            if not os.path.exists(aug_mel_path):
                np.save(aug_mel_path,np.asanyarray((aug_mel,keyshift),dtype=object))
        if not os.path.exists(aug_vol_path):
            aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
            np.save(aug_vol_path,aug_vol.to('cpu').numpy())

    if c is None:
        c = torch.load(soft_path)
    if f0 is None or uv is None:
        f0, uv = np.load(f0_path, allow_pickle=True)
    if spec is None:
        spec = torch.load(spec_path)
    if hps.model.vol_embedding and volume is None:
        volume_path = filename + ".vol.npy"
        if os.path.exists(volume_path):
            volume = np.load(volume_path)

    train_bundle = {
        "version": 1,
        "audio": audio_norm.cpu(),
        "soft": c.cpu() if isinstance(c, torch.Tensor) else torch.from_numpy(np.asarray(c)),
        "f0": torch.FloatTensor(np.array(f0, dtype=float)),
        "uv": torch.FloatTensor(np.array(uv, dtype=float)),
        "spec": spec.cpu() if isinstance(spec, torch.Tensor) else torch.from_numpy(np.asarray(spec)),
        "volume": None if volume is None else (
            volume.cpu() if isinstance(volume, torch.Tensor) else torch.from_numpy(np.asarray(volume)).float()
        ),
    }
    torch.save(train_bundle, train_bundle_path)


def process_batch(file_chunk, f0p, diff=False, device="cpu", config_path="configs/config.json", diff_config_path="configs/diffusion.yaml"):
    if hps is None or dconfig is None:
        load_runtime_configs(config_path, diff_config_path)
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    logger.info(f"Loaded speech encoder for rank {rank}")
    mel_extractor = None
    if diff:
        mel_extractor = Vocoder(dconfig.vocoder.type, dconfig.vocoder.ckpt, device=device)
        logger.info(f"Loaded mel extractor for rank {rank}")
    for filename in tqdm(file_chunk, position = rank):
        process_one(filename, hmodel, f0p, device, diff, mel_extractor)

def parallel_process(filenames, num_processes, f0p, diff, device, config_path, diff_config_path):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(
                executor.submit(
                    process_batch,
                    file_chunk,
                    f0p,
                    diff,
                    device=device,
                    config_path=config_path,
                    diff_config_path=diff_config_path,
                )
            )
        for task in tqdm(tasks, position = 0):
            task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.json", help="path to main config"
    )
    parser.add_argument(
        "--diff_config", type=str, default="configs/diffusion.yaml", help="path to diffusion config"
    )
    # 极致音质配置：默认启用浅层扩散
    parser.add_argument(
        '--use_diff', action='store_true', help='[极致音质配置] 启用浅层扩散（默认已启用）'
    )
    parser.add_argument(
        '--no_use_diff', action='store_true', help='[极致音质配置] 禁用浅层扩散'
    )
    # 极致音质配置：固定使用 rmvpe F0预测器
    parser.add_argument(
        '--f0_predictor', type=str, default="rmvpe", help='[极致音质配置] 固定使用 rmvpe F0预测器'
    )
    parser.add_argument(
        '--num_processes', type=int, default=1, help='You are advised to set the number of processes to the same as the number of CPU cores'
    )
    args = parser.parse_args()
    load_runtime_configs(args.config, args.diff_config)
    f0p = args.f0_predictor
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 极致音质配置：默认启用浅层扩散，除非明确禁用
    use_diff = not args.no_use_diff

    logger.info("Using device: " + str(device))
    logger.info("Using SpeechEncoder: " + speech_encoder)
    logger.info("Using extractor: " + f0p)
    logger.info("Using diff Mode: " + str(use_diff))

    if use_diff:
        logger.info("Mel extractor will be initialized per worker process.")
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    parallel_process(filenames, num_processes, f0p, use_diff, device, args.config, args.diff_config)
