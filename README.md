[![PyPI version](https://badge.fury.io/py/kvpress.svg)](https://badge.fury.io/py/kvpress)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Colab example notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP?usp=drive_link)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/nvidia/kvpress)
[![Blog post](https://img.shields.io/badge/🤗%20Hugging%20Face-Blog-blue)](https://huggingface.co/blog/nvidia/kvpress)
[![Hugging Face Leaderboard](https://img.shields.io/badge/🤗%20HuggingFace-Leaderboard-orange)](https://huggingface.co/spaces/nvidia/kvpress-leaderboard)

![kvpress](kvpress.jpg)


Deploying long-context LLMs is costly due to the linear growth of the key-value (KV) cache in transformer models. For example, handling 1M tokens with Llama 3.1-70B in float16 requires up to 330GB of memory. kvpress implements multiple KV cache compression methods and benchmarks using 🤗 transformers, aiming to simplify the development of new methods for researchers and developers in this field.
## LU-KV
This repository is a fork of **[NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)**. It implements **LU-KV**, a framework that optimizes head-wise KV cache budget allocation by maximizing the long-horizon marginal utility of tokens, as proposed in our paper: "[Predicting Future Utility: Global Combinatorial Optimization for Task-Agnostic KV Cache Eviction](https://arxiv.org/pdf/2602.08585)".

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
