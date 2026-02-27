import logging
from dataclasses import dataclass
import torch
from torch import nn
from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)

@dataclass
class PyramidKVWrap(BasePress):
    """
    PyramidKV 作为包装器的实现。
    它可以包装任何提供 score() 方法的 ScorerPress。
    """
    press: ScorerPress  # 被包装的打分器，例如 SnapKVPress
    beta: int = 20
    window_size: int = 64

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "PyramidKVPress 必须包装一个 ScorerPress"
        assert self.beta >= 1, "Beta 应该 >= 1"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def get_layer_budget(self, module: nn.Module, q_len: int) -> int:
        """
        金字塔预算计算逻辑（保持不变）
        """
        # 计算该层应保留的长度 n_kept
        max_capacity_prompt = self.window_size + q_len * (1 - self.compression_ratio)
        min_num = (max_capacity_prompt - self.window_size) / self.beta
        max_num = (max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (max_capacity_prompt - self.window_size) * 2 - max_num

        if not (q_len >= max_num >= min_num >= self.window_size):
            return round(q_len * (1 - self.compression_ratio))

        steps = (max_num - min_num) / (module.config.num_hidden_layers - 1)
        return round(max_num - module.layer_idx * steps)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values

        # 核心逻辑：调用被包装压榨器的 score 方法
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        # 根据金字塔逻辑计算当前层的 Budget
        q_len = hidden_states.shape[1]
        n_kept = self.get_layer_budget(module, q_len)

        # 执行 Top-k 筛选
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values