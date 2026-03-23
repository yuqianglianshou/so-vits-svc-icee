from __future__ import annotations
"""推理页音频转换与结果整理辅助函数。

这里集中放上传音频、TTS 音频和高质量预设转换链，
让 app_infer.py 入口文件只保留页面事件和全局模型状态。
"""

import os
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile

from infer_ui.runtime import get_model_device_name
from infer_ui.text import build_runtime_summary, render_convert_result_html



def vc_infer_with_model(model, output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    """调用已加载模型执行切片推理并写出结果文件。"""
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
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment,
    )
    model.clear_empty()
    os.makedirs("results", exist_ok=True)
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    diffusion_tag = "sovits"
    if model.shallow_diffusion:
        diffusion_tag = "sovdiff"
    if model.only_diffusion:
        diffusion_tag = "diff"
    output_file_name = f"result_{truncated_basename}_{sid}_{key}{cluster}{diffusion_tag}.{output_format}"
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, audio, model.target_sample, format=output_format)
    return output_file



def convert_uploaded_audio(model, sid, input_audio, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    """处理上传音频并调用模型完成转换。"""
    if input_audio is None:
        return "You need to upload an audio", None
    if model is None:
        return "You need to upload an model", None
    if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False and cluster_ratio != 0:
        return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None

    audio, sampling_rate = soundfile.read(input_audio)
    if np.issubdtype(audio.dtype, np.integer):
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    truncated_basename = Path(input_audio).stem[:-6]
    processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
    soundfile.write(processed_audio, audio, sampling_rate, format="wav")
    output_file = vc_infer_with_model(
        model,
        output_format,
        sid,
        processed_audio,
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
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment,
    )
    return "Success", output_file



def convert_tts_audio(model, text, lang, gender, rate, volume, sid, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    """先生成 TTS，再走同一条推理转换链。"""
    if model is None:
        return "You need to upload an model", None
    if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False and cluster_ratio != 0:
        return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None

    rate = f"+{int(rate*100)}%" if rate >= 0 else f"{int(rate*100)}%"
    volume = f"+{int(volume*100)}%" if volume >= 0 else f"{int(volume*100)}%"
    if lang == "Auto":
        gender = "Male" if gender == "男" else "Female"
        subprocess.run([sys.executable, "edgetts/tts.py", text, lang, rate, volume, gender])
    else:
        subprocess.run([sys.executable, "edgetts/tts.py", text, lang, rate, volume])
    target_sr = 44100
    y, sr = librosa.load("tts.wav")
    resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    soundfile.write("tts.wav", resampled_y, target_sr, subtype="PCM_16")
    output_file_path = vc_infer_with_model(
        model,
        output_format,
        sid,
        "tts.wav",
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
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment,
    )
    os.remove("tts.wav")
    return "Success", output_file_path



def quality_convert(model, sid, input_audio, quality_mode, vc_transform, cluster_ratio, k_step, best_quality_preset):
    """按高质量预设执行一次转换，并整理结果文案。"""
    preflight = ""
    if model is not None:
        if not model.shallow_diffusion:
            preflight += "提醒：当前没有加载音质增强模型，转换可以继续，但不会是最佳音质。\n"
        if getattr(model, "cluster_model", None) is None:
            preflight += "提醒：当前没有加载音色增强文件，音色相似度可能低于最佳状态。\n"
    msg, audio = convert_uploaded_audio(
        model,
        sid,
        input_audio,
        best_quality_preset["output_format"],
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
        best_quality_preset["use_spk_mix"],
        best_quality_preset["second_encoding"],
        best_quality_preset["loudness_envelope_adjustment"],
    )
    if audio is None:
        return render_convert_result_html(preflight + str(msg)), audio
    runtime_summary = build_runtime_summary(
        get_model_device_name(model),
        sid,
        quality_mode,
        vc_transform,
        cluster_ratio,
        k_step,
        bool(getattr(model, "shallow_diffusion", False) or getattr(model, "only_diffusion", False)),
        getattr(model, "cluster_model", None) is not None,
    )
    result_msg = f"{preflight}转换完成\n{runtime_summary}\n输出文件：{audio}"
    return render_convert_result_html(result_msg), audio
