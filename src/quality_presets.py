BEST_QUALITY_PRESET = {
    "output_format": "flac",
    "auto_predict_f0": False,
    "f0_predictor": "rmvpe",
    "cluster_ratio": 0.5,
    "slice_db": -40,
    "noise_scale": 0.4,
    "k_step": 200,
    "pad_seconds": 0.5,
    "clip_seconds": 0,
    "linear_gradient": 0,
    "linear_gradient_retain": 0.75,
    "enhancer_adaptive_key": 0,
    "cr_threshold": 0.05,
    "loudness_envelope_adjustment": 1.0,
    "second_encoding": True,
    "enhance": False,
    "only_diffusion": False,
}


QUALITY_MODES = {
    "标准高质": {
        "cluster_ratio": 0.35,
        "k_step": 120,
    },
    "极致质量": {
        "cluster_ratio": 0.5,
        "k_step": 200,
    },
}
