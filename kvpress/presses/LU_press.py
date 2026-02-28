# kvpress/presses/lu_press.py
import logging
from dataclasses import dataclass, field
from typing import Optional, Generator
from contextlib import contextmanager

import torch
from torch import nn
import numpy as np

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)

@dataclass
class LUPress(BasePress):
    press: ScorerPress            
    budget_curve_path: Optional[str] = None  
    
    sink: int = 4       
    window: int = 32     

    _budget_curves: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    _global_kept_tokens: int = field(init=False, repr=False, default=0)
    _global_total_tokens: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "Only support `ScorerPress`"


    def _post_setup_init(self):
        if self.budget_curve_path:
            try:
                print(f"Wrapped Press: {self.press.__class__.__name__}")
                print(f"Budget Curves Path: {self.budget_curve_path}")
                
                data = np.load(self.budget_curve_path)
                self._budget_curves = data
                print(f"Sink={self.sink}, Window={self.window}\n")
            except Exception as e:
                raise IOError(f"Loading Budget Curves Failed: {e}")
        else:
            print("No Budget Curves Loaded")

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    @contextmanager
    def __call__(self, model: nn.Module) -> Generator:
        self._global_kept_tokens = 0
        self._global_total_tokens = 0
        with super().__call__(model):
            yield

    def compress(self, module: nn.Module, hidden_states: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attentions: torch.Tensor, kwargs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio <= 0: return keys, values
        bsz, num_heads, seq_len, _ = keys.shape
        if self._budget_curves is None: return keys, values 

        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        if self.sink > 0:
            safe_sink = min(self.sink, seq_len)
            scores[..., :safe_sink] = scores.max().item()
        if self.window > 0:
            start_idx = max(0, seq_len - self.window)
            scores[..., start_idx:] = scores.max().item()

        target_idx = int(round(self.compression_ratio * 100)) - 1
        target_idx = max(0, min(98, target_idx))
        layer_idx = module.layer_idx
        
        try:
            local_prune_ratios = torch.from_numpy(self._budget_curves[target_idx, layer_idx]).to(keys.device)
        except IndexError:
            return keys, values

        head_keep_rates = 1.0 - local_prune_ratios
        ideal_keep_counts = head_keep_rates * seq_len
        total_keep_target = int(torch.round(torch.sum(ideal_keep_counts)).item())
        base_keep_counts = torch.floor(ideal_keep_counts).long()
        remainder = total_keep_target - base_keep_counts.sum()

        if remainder > 0:
            fractional_parts = ideal_keep_counts - base_keep_counts
            num_to_distribute = min(int(remainder.item()), num_heads)
            if num_to_distribute > 0:
                top_k_indices = torch.topk(fractional_parts, k=num_to_distribute).indices
                base_keep_counts[top_k_indices] += 1
        
        final_keep_per_head = base_keep_counts.clamp_(min=1, max=seq_len)
        num_to_prune_per_head = seq_len - final_keep_per_head


        current_layer_kept = final_keep_per_head.sum().item()
        current_layer_total = seq_len * num_heads
        self._global_kept_tokens += current_layer_kept
        self._global_total_tokens += current_layer_total

        total_layers = getattr(module.config, "num_hidden_layers", None)
        if total_layers is not None and layer_idx == total_layers - 1:
            total_ratio = self._global_kept_tokens / self._global_total_tokens
            print(f"[LUPress] Final Global Keep Ratio: {total_ratio:.6f} (Target: {1.0 - self.compression_ratio:.2f})")

        if torch.all(num_to_prune_per_head <= 0):
            if hasattr(module, 'masked_key_indices'):
                module.masked_key_indices = None
            return keys, values
            
        sorted_indices = torch.argsort(scores.squeeze(0), dim=-1, descending=True, stable=True)
        
        rank = torch.arange(seq_len, device=scores.device).expand_as(sorted_indices)
        keep_mask = rank < final_keep_per_head.unsqueeze(1)
        
        prune_mask = ~keep_mask

        pruned_seq_indices = sorted_indices[prune_mask]
        head_indices = torch.arange(num_heads, device=scores.device).unsqueeze(1).expand_as(sorted_indices)[prune_mask]
        batch_indices = torch.zeros_like(head_indices)

        module.masked_key_indices = (batch_indices, head_indices, pruned_seq_indices)

        return keys, values