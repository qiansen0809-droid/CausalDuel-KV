# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
from transformers.models.phi3.modeling_phi3 import Phi3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class SnapKVPress(ScorerPress):
    """
    SnapKV: Attention-based KV cache compression using recent token patterns.
    """

    compression_ratio: float = 0.0
    window_size: int = 512
    kernel_size: int = 5
    n_sink: int = 4  # sink tokens 数量
    
    # 保存相关参数
    dataset: str = None
    data_dir: str = None
    model_name: str = None
    scores_output_filepath: str = None
    
    # internal state
    _sample_id: int = field(default=0, init=False, repr=False)
    _collected_scores: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._post_setup_init()

    def _post_setup_init(self):
        if self.model_name and self.dataset and self.data_dir:
            if not self.scores_output_filepath:
                self.scores_output_filepath = f"/ssd2/tangziyao/code/data_generate_dataset/{self.model_name}/snapkv_{self.dataset}_output/{self.data_dir}_output"
        logger.info(f"SnapKVPress initialized. scores_output: {self.scores_output_filepath}")

    def _save_scores_as_npy(self):
        """
        保存 scores 为 npy 文件，格式与 KVzipPress 一致。
        只保存真正有 scores 的部分（去掉 sink 和 recent window）。
        Shape: (n_layer, num_kv_heads, ctx_len - n_sink - window_size)
        """
        if not self.scores_output_filepath:
            return
        if not self._collected_scores:
            logger.warning(f"sample {self._sample_id} 没有收集到 scores，跳过保存")
            return
        try:
            out_dir = Path(self.scores_output_filepath)
            out_dir.mkdir(parents=True, exist_ok=True)
            file_path = out_dir / f"sample_{self._sample_id}.npy"
            
            # 按 layer 顺序组装
            n_layers = max(self._collected_scores.keys()) + 1
            first_score = next(iter(self._collected_scores.values()))
            num_kv_heads = first_score.shape[1]
            # 只取有效部分的长度
            valid_len = first_score.shape[2]
            
            score_array = torch.zeros((n_layers, num_kv_heads, valid_len), dtype=torch.float16)
            for layer_idx, score in self._collected_scores.items():
                # score shape: (bsz, num_kv_heads, valid_len), 取 bsz=0
                score_array[layer_idx] = score[0].cpu().to(torch.float16)
            
            np.save(file_path, score_array.numpy())
            logger.info(f"Saved score npy for sample {self._sample_id} to {file_path}, shape: {score_array.shape}")
        except Exception as e:
            logger.error(f"Failed to save score npy for sample {self._sample_id}: {e}")

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """

        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # Get last window_size queries
        if isinstance(module, Phi3Attention):
            qkv = module.qkv_proj(hidden_states[:, -window_size:])
            query_states = qkv[..., : num_heads * head_dim]
        elif hasattr(module, "q_proj"):
            # Assume Llama-like attention layer
            query_states = module.q_proj(hidden_states[:, -window_size:])
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

        # Support for Qwen3 and Gemma3 QK norm
        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
            query_states = module.q_norm(query_states)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert q_len > self.window_size, "Query length should be greater than the window size"

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        # 计算原始的 max scores（不做平滑）
        # attn_weights shape: (bsz, num_heads, window_size, q_len - window_size)

        # 先 reshape 成 group 形式
        ctx_len = q_len - self.window_size
        attn_weights_grouped = attn_weights.view(bsz, num_key_value_heads, num_key_value_groups, self.window_size, ctx_len)

        # 在 group 维度和 window 维度上取最大值（和 kvzip 一样）
        raw_max_scores = attn_weights_grouped.amax(dim=(2, 3))
        # raw_max_scores shape: (bsz, num_kv_heads, ctx_len)

        # 只保存有效部分：去掉前 n_sink 个 token
        # raw_max_scores 已经不包含 window 部分了，只需要去掉 sink
        valid_scores = raw_max_scores[:, :, self.n_sink:]
        # valid_scores shape: (bsz, num_kv_heads, ctx_len - n_sink)

        # 保存原始 scores（只保存有效部分）
        layer_idx = module.layer_idx
        self._collected_scores[layer_idx] = valid_scores.detach().clone()

        # 原始 SnapKV 的 scores 计算（用于压缩）
        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        # Average per group
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
        scores = scores.mean(2)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        return scores

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """重写 context manager，每次退出时保存当前 sample 的 scores"""
        with super().__call__(model):
            try:
                yield
            finally:
                # 保存当前 sample 的 scores
                self._save_scores_as_npy()
                self._sample_id += 1
                self._collected_scores = {}