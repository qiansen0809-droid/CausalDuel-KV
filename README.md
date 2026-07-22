[![PyPI version](https://badge.fury.io/py/kvpress.svg)](https://badge.fury.io/py/kvpress)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Colab example notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP?usp=drive_link)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/nvidia/kvpress)
[![Blog post](https://img.shields.io/badge/🤗%20Hugging%20Face-Blog-blue)](https://huggingface.co/blog/nvidia/kvpress)
[![Hugging Face Leaderboard](https://img.shields.io/badge/🤗%20HuggingFace-Leaderboard-orange)](https://huggingface.co/spaces/nvidia/kvpress-leaderboard)

![kvpress](kvpress.jpg)

Deploying long-context LLMs is costly due to the linear growth of the key-value (KV) cache in transformer models. For example, handling 1M tokens with Llama 3.1-70B in float16 requires up to 330GB of memory. kvpress implements multiple KV cache compression methods and benchmarks using 🤗 transformers, aiming to simplify the development of new methods for researchers and developers in this field.

## LU-KV

This repository is a fork of **[NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)**. It implements **LU-KV**, a framework that optimizes head-wise KV cache budget allocation by maximizing the long-horizon marginal utility of tokens, as proposed in our paper: "[Predicting Future Utility: Global Combinatorial Optimization for Task-Agnostic KV Cache Eviction](https://arxiv.org/abs/2602.08585)".

> 🎉 **Accepted at ICML 2026.** [[Paper](https://arxiv.org/abs/2602.08585)] [[OpenReview](https://openreview.net/forum?id=FQLxcBsKIb)] [[ICML](https://icml.cc/virtual/2026/poster/65241)]

### How LU-KV Works

KV cache eviction involves two decisions: selecting important tokens within each attention head and distributing the global cache budget across heads. Existing allocation methods often compare instantaneous heuristic scores across heads, even though their scales and ability to predict future token utility can differ substantially.

LU-KV instead allocates the cache budget according to the **long-horizon marginal utility** of each attention head. Given a global budget \(B\), it minimizes the total long-horizon eviction loss:

$$
\min_{\{b_h\}} \sum_h \mathcal{L}_h(b_h)
\quad \text{s.t.} \quad
\sum_h b_h = B,
$$

where $b_h$ is the budget assigned to head $h$, and $\mathcal{L}_h(b_h)$ is its long-horizon eviction loss.

1. **Long-horizon utility profiling.** LU-KV uses future full-attention decoding offline to estimate Oracle Importance and construct a loss-versus-budget curve for each attention head.
2. **Global budget optimization.** Because the discrete loss curves can be non-convex, LU-KV applies convex-hull relaxation and a marginal-utility-based greedy solver to obtain a near-optimal head-wise allocation.
3. **Efficient online execution.** The optimized allocations are aggregated into a static lookup table. During inference, LU-KV only looks up the per-head budgets, while the underlying metric—such as SnapKV, KeyDiff, or EA—selects tokens within each head.

LU-KV is therefore a metric-agnostic budget-allocation layer rather than a replacement for existing token-scoring methods. On LongBench and RULER, it reduces KV cache size by 80% with minimal performance degradation while also reducing inference latency and GPU memory usage.

## Installation

```bash
cd LU-KV
pip install -e .
pip install transformers==4.53.0 fastparquet rouge nltk jieba fuzzywuzzy bert-score fire seaborn scikit-learn argparse jieba fuzzywuzzy rouge accelerate sentencepiece datasets wandb zstandard matplotlib huggingface_hub flash-attn
```

## Usage

### Evaluation

Run LU-KV (all three variants: lu_snapkv → lu_keydiff → lu_ea) on LongBench and RULER:

```bash
cd evaluation
bash lukv_longbench.sh   # LongBench (16 tasks)
bash lukv_ruler.sh       # RULER
```

Run all baseline methods (SnapKV, KeyDiff, EA and their AdaKV / PyramidKV variants):

```bash
cd evaluation
bash baseline_longbench.sh
bash baseline_ruler.sh
```

### Save Curve Data

```bash
cd evaluation
bash curve_data/generate_curve.sh
```

## Citation

If you find LU-KV useful in your research, please cite our paper:

```bibtex
@inproceedings{tang2026predicting,
  title     = {Predicting Future Utility: Global Combinatorial Optimization for Task-Agnostic {KV} Cache Eviction},
  author    = {Tang, Ziyao and Jiao, Pengkun and Chen, Xinhang and Liu, Wei and Li, Shiyong and Chen, Jingjing},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=FQLxcBsKIb}
}
```
