"""项目当前唯一保留的 ContentVec 实现，底层使用 Transformers/HF 路线。"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import HubertModel

from src.path_utils import get_contentvec_hf_path
from src.vencoder.encoder import SpeechEncoder


DEFAULT_HF_CONTENTVEC_REPO = "lengyue233/content-vec-best"


class HubertModelWithFinalProj(HubertModel):
    """兼容 Hugging Face 上转换后的 ContentVec 权重。"""

    def __init__(self, config):
        super().__init__(config)
        classifier_proj_size = getattr(config, "classifier_proj_size", config.hidden_size)
        self.final_proj = nn.Linear(config.hidden_size, classifier_proj_size)


class ContentVec768L12(SpeechEncoder):
    """加载 Hugging Face 兼容权重，并输出第 12 层 768 维内容特征。"""

    def __init__(self, vec_path=None, device=None):
        super().__init__()
        self.hidden_dim = 768
        self.dev = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_ref = self._resolve_model_ref(vec_path)
        print(f"load model(s) from {self.model_ref}")
        self.model = HubertModelWithFinalProj.from_pretrained(
            self.model_ref,
            use_safetensors=True,
        ).to(self.dev)
        self.model.eval()

    def _resolve_model_ref(self, vec_path):
        """解析 HF 模型来源，并对旧版 fairseq 权重给出明确提示。"""
        if vec_path is not None:
            candidate = str(vec_path)
        else:
            candidate = str(get_contentvec_hf_path())

        candidate_path = Path(candidate)
        if candidate_path.suffix == ".pt":
            raise ValueError(
                "当前项目的 ContentVec 已切换到 transformers/HF 路线，不能再直接加载 fairseq .pt 权重。"
                "请提供 Hugging Face 模型目录、模型仓库名，"
                "或先将兼容模型放到 model_assets/dependencies/encoders/contentvec_hf。"
            )
        if candidate_path.exists():
            return str(candidate_path)
        if (
            "/" in candidate
            and not candidate_path.is_absolute()
            and not candidate.startswith(".")
            and not candidate.startswith("~")
            and not candidate_path.exists()
        ):
            return candidate
        fallback = str(get_contentvec_hf_path())
        if Path(fallback).exists():
            return fallback
        return DEFAULT_HF_CONTENTVEC_REPO

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        inputs = feats.view(1, -1).to(self.dev)
        attention_mask = torch.ones_like(inputs, dtype=torch.long, device=self.dev)
        with torch.no_grad():
            outputs = self.model(
                input_values=inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if not hidden_states or len(hidden_states) < 13:
            raise RuntimeError("当前 HF 模型没有足够的 hidden_states，无法提取第 12 层特征。")
        features = hidden_states[12]
        if features.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"HF ContentVec 输出维度为 {features.shape[-1]}，预期为 {self.hidden_dim}。"
            )
        return features.transpose(1, 2)
