from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile

from src.infer_ui.runtime import get_model_device_name
from src.infer_ui.text import render_convert_result_html


def write_output_audio(output_file: str, audio, sample_rate: int, output_format: str):
    output_format = (output_format or "flac").lower()
    if output_format in {"flac", "wav"}:
        soundfile.write(output_file, audio, sample_rate, format=output_format)
        return output_file
    if output_format == "mp3":
        wav_temp_path = str(Path(output_file).with_suffix(".mp3_tmp.wav"))
        soundfile.write(wav_temp_path, audio, sample_rate, format="wav")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    wav_temp_path,
                    "-codec:a",
                    "libmp3lame",
                    "-q:a",
                    "2",
                    output_file,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("当前环境缺少 ffmpeg，无法导出 mp3。") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("ffmpeg 导出 mp3 失败。") from exc
        finally:
            if os.path.exists(wav_temp_path):
                os.remove(wav_temp_path)
        return output_file
    raise ValueError(f"不支持的输出格式：{output_format}")


def write_preview_and_download_audio(base_output_stem: str, audio, sample_rate: int):
    """固定生成 flac 试听文件，并同时提供 flac / wav 两种下载文件。"""
    flac_file = f"{base_output_stem}.flac"
    wav_file = f"{base_output_stem}.wav"
    write_output_audio(flac_file, audio, sample_rate, "flac")
    write_output_audio(wav_file, audio, sample_rate, "wav")
    return flac_file, flac_file, wav_file


def vc_infer_with_model(
    model,
    sid,
    audio_path,
    truncated_basename,
    vc_transform,
    auto_f0,
    cluster_ratio,
    slice_db,
    noise_scale,
    pad_seconds,
    cl_num,
    lg_num,
    lgr_num,
    f0_predictor,
    enhancer_adaptive_key,
    cr_threshold,
    k_step,
    second_encoding,
    loudness_envelope_adjustment,
):
    audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        second_encoding,
        loudness_envelope_adjustment,
    )
    model.clear_empty()
    os.makedirs("inference_data/outputs", exist_ok=True)
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    diffusion_tag = "sovits"
    if model.shallow_diffusion:
        diffusion_tag = "sovdiff"
    if model.only_diffusion:
        diffusion_tag = "diff"
    output_stem = os.path.join("inference_data/outputs", f"result_{truncated_basename}_{sid}_{key}{cluster}{diffusion_tag}")
    return write_preview_and_download_audio(output_stem, audio, model.target_sample)


def convert_uploaded_audio(
    model,
    sid,
    input_audio,
    vc_transform,
    auto_f0,
    cluster_ratio,
    slice_db,
    noise_scale,
    pad_seconds,
    cl_num,
    lg_num,
    lgr_num,
    f0_predictor,
    enhancer_adaptive_key,
    cr_threshold,
    k_step,
    second_encoding,
    loudness_envelope_adjustment,
):
    if input_audio is None:
        return "You need to upload an audio", None, None, None
    if model is None:
        return "You need to upload an model", None, None, None
    if getattr(model, "cluster_model", None) is None and model.feature_retrieval is False and cluster_ratio != 0:
        cluster_ratio = 0

    audio, sampling_rate = soundfile.read(input_audio)
    if np.issubdtype(audio.dtype, np.integer):
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    truncated_basename = Path(input_audio).stem
    temp_input = tempfile.NamedTemporaryFile(prefix="infer_input_", suffix=".wav", delete=False)
    temp_input.close()
    try:
        soundfile.write(temp_input.name, audio, sampling_rate, format="wav")
        preview_file, flac_download_file, wav_download_file = vc_infer_with_model(
            model,
            sid,
            temp_input.name,
            truncated_basename,
            vc_transform,
            auto_f0,
            cluster_ratio,
            slice_db,
            noise_scale,
            pad_seconds,
            cl_num,
            lg_num,
            lgr_num,
            f0_predictor,
            enhancer_adaptive_key,
            cr_threshold,
            k_step,
            second_encoding,
            loudness_envelope_adjustment,
        )
        return "Success", preview_file, flac_download_file, wav_download_file
    finally:
        if os.path.exists(temp_input.name):
            os.remove(temp_input.name)


def convert_tts_audio(
    model,
    text,
    lang,
    gender,
    rate,
    volume,
    sid,
    vc_transform,
    auto_f0,
    cluster_ratio,
    slice_db,
    noise_scale,
    pad_seconds,
    cl_num,
    lg_num,
    lgr_num,
    f0_predictor,
    enhancer_adaptive_key,
    cr_threshold,
    k_step,
    second_encoding,
    loudness_envelope_adjustment,
):
    if model is None:
        return "You need to upload an model", None, None, None
    if getattr(model, "cluster_model", None) is None and model.feature_retrieval is False and cluster_ratio != 0:
        cluster_ratio = 0

    rate = f"+{int(rate * 100)}%" if rate >= 0 else f"{int(rate * 100)}%"
    volume = f"+{int(volume * 100)}%" if volume >= 0 else f"{int(volume * 100)}%"
    gender = "Male" if gender == "男" else "Female"
    temp_tts = tempfile.NamedTemporaryFile(prefix="infer_tts_", suffix=".wav", delete=False)
    temp_tts.close()
    try:
        subprocess.run(
            [sys.executable, "-m", "src.edgetts.tts", text, lang, rate, volume, gender, temp_tts.name],
            check=True,
        )
        target_sr = 44100
        y, sr = librosa.load(temp_tts.name)
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write(temp_tts.name, resampled_y, target_sr, subtype="PCM_16")
        preview_file, flac_download_file, wav_download_file = vc_infer_with_model(
            model,
            sid,
            temp_tts.name,
            "tts",
            vc_transform,
            auto_f0,
            cluster_ratio,
            slice_db,
            noise_scale,
            pad_seconds,
            cl_num,
            lg_num,
            lgr_num,
            f0_predictor,
            enhancer_adaptive_key,
            cr_threshold,
            k_step,
            second_encoding,
            loudness_envelope_adjustment,
        )
        return "Success", preview_file, flac_download_file, wav_download_file
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"文本转语音失败，退出码 {exc.returncode}。") from exc
    finally:
        if os.path.exists(temp_tts.name):
            os.remove(temp_tts.name)


def quality_convert(model, sid, input_audio, quality_mode, vc_transform, cluster_ratio, k_step, best_quality_preset):
    warnings = []
    if model is not None:
        if not model.shallow_diffusion:
            warnings.append("未加载音质增强")
        if getattr(model, "cluster_model", None) is None:
            warnings.append("未加载音色增强")
            if cluster_ratio != 0:
                cluster_ratio = 0

    msg, preview_audio, flac_download_audio, wav_download_audio = convert_uploaded_audio(
        model,
        sid,
        input_audio,
        vc_transform,
        best_quality_preset["auto_predict_f0"],
        cluster_ratio,
        best_quality_preset["slice_db"],
        best_quality_preset["noise_scale"],
        best_quality_preset["pad_seconds"],
        best_quality_preset["clip_seconds"],
        best_quality_preset["linear_gradient"],
        best_quality_preset["linear_gradient_retain"],
        best_quality_preset["f0_predictor"],
        best_quality_preset["enhancer_adaptive_key"],
        best_quality_preset["cr_threshold"],
        k_step,
        best_quality_preset["second_encoding"],
        best_quality_preset["loudness_envelope_adjustment"],
    )
    if preview_audio is None:
        return render_convert_result_html(str(msg)), preview_audio, flac_download_audio, wav_download_audio

    summary_lines = [
        "转换完成",
        f"设备：{get_model_device_name(model)}",
        f"目标音色：{sid}",
        f"变调：{vc_transform}",
    ]
    if warnings:
        summary_lines.append("当前链路：" + "、".join(warnings))
    return render_convert_result_html("\n".join(summary_lines)), preview_audio, flac_download_audio, wav_download_audio
