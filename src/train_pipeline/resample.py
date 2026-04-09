import argparse
import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import librosa
import numpy as np
from rich.progress import track
from scipy.io import wavfile


def load_wav(wav_path):
    return librosa.load(wav_path, sr=None)


def trim_wav(wav, top_db=40):
    return librosa.effects.trim(wav, top_db=top_db)


def resample_wav(wav, sr, target_sr):
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


def save_wav_to_path(wav, save_path, sr):
    wavfile.write(
        save_path,
        sr,
        (wav * np.iinfo(np.int16).max).astype(np.int16)
    )


def process(item):
    spkdir, wav_name, args = item
    speaker = spkdir.replace("\\", "/").split("/")[-1]
    out_dir = resolve_output_dir(args.out_dir2, speaker, bool(args.speaker))

    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and wav_name.lower().endswith(".wav"):
        os.makedirs(out_dir, exist_ok=True)

        wav, sr = load_wav(wav_path)
        wav, _ = trim_wav(wav)
        resampled_wav = resample_wav(wav, sr, args.sr2)

        save_path2 = os.path.join(out_dir, wav_name)
        save_wav_to_path(resampled_wav, save_path2, args.sr2)


def resolve_output_dir(out_dir: str, speaker: str, selected_single_speaker: bool) -> str:
    """兼容两种输出模式：
    1. UI 主线：out_dir 已经是 training_data/processed/44k/<speaker>，不再重复套一层；
    2. 旧 CLI：out_dir 指向父目录时，仍保留 <speaker>/ 子目录。
    """
    out_path = Path(out_dir)
    if selected_single_speaker and out_path.name == speaker:
        return str(out_path)
    return str(out_path / speaker)


"""
def process_all_speakers():
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)

    with ThreadPoolExecutor(max_workers=process_count) as executor:
        for speaker in speakers:
            spk_dir = os.path.join(args.in_dir, speaker)
            if os.path.isdir(spk_dir):
                print(spk_dir)
                futures = [executor.submit(process, (spk_dir, i, args)) for i in os.listdir(spk_dir) if i.endswith("wav")]
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass
"""
# multi process


def process_all_speakers():
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for speaker in speakers:
            spk_dir = os.path.join(args.in_dir, speaker)
            if os.path.isdir(spk_dir):
                print(spk_dir)
                futures = [executor.submit(process, (spk_dir, i, args)) for i in os.listdir(spk_dir) if i.lower().endswith(".wav")]
                for future in track(concurrent.futures.as_completed(futures), total=len(futures), description="resampling:"):
                    future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr2", type=int, default=44100, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="./training_data/source", help="path to source dir")
    parser.add_argument("--speaker", type=str, default="", help="only process the selected speaker folder under in_dir")
    parser.add_argument("--out_dir2", type=str, default="./training_data/processed/44k", help="path to target dir")
    args = parser.parse_args()

    print(f"CPU count: {cpu_count()}")
    if args.speaker:
        speakers = [args.speaker]
    else:
        speakers = os.listdir(args.in_dir)
    process_all_speakers()
