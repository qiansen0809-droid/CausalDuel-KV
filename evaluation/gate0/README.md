# CausalDuel-KV Gate 0 POC

This branch implements the first scientific feasibility check around the official LU-KV codebase.

## Goal

Before building cache-fork, page sharing, successive racing, or serving integration, test one narrow question:

> Starting from the LU-KV allocation, does a small budget-preserving donor/receiver swap that improves prompt-tail behavior also improve a teacher-forced continuation objective?

The first POC intentionally validates the algorithmic signal using LU-KV's existing logical head masking. It does **not** claim physical ragged-cache memory savings. Physical page/block validation remains Gate -1.

## What changed

### 1. Exact per-KV-head budget override

`kvpress/presses/LU_press.py` now accepts:

```python
keep_counts_override: dict[int, list[int]]
```

The mapping is:

```text
layer_idx -> [keep_count_head0, keep_count_head1, ...]
```

If a layer is overridden, the explicit keep counts replace the LU-KV static profile for that layer. The wrapped scorer is unchanged, so Gate 0 varies only budget allocation.

The implementation validates:

- one count per runtime KV head;
- donor budgets cannot go below the protected sink + recent window;
- receiver budgets cannot exceed prefix length;
- batch size is 1 for the current head-wise masking path.

The exact counts used during a prefill are exposed through `last_keep_counts`.

### 2. Budget-preserving swap construction

`build_candidates.py` provides a small donor/receiver representation and checks that every swap preserves the exact global keep count.

For the engineering POC, the fallback ordering chooses large-budget heads as donors and small-budget heads as receivers. This is only a smoke-test heuristic.

**For the actual MiniGate, replace this fallback with LU-KV remove-cost / next-page-gain rankings from the offline utility curves.**

### 3. Prompt-tail replay

`suffix_replay.py`:

1. prefills the prefix;
2. optionally applies LU-KV / an overridden LU-KV budget;
3. teacher-forces the known prompt suffix;
4. returns suffix logits.

The same suffix is replayed for FullKV, LU-KV baseline, and every swap candidate.

### 4. Metrics

`metrics.py` currently implements:

- full-vocabulary probability overlap;
- teacher-to-candidate KL;
- Jensen-Shannon divergence;
- top-k agreement;
- teacher-forced continuation NLL.

For the first POC, suffix-token NLL is used as a dense sanity target. The registered Gate 0 design still requires **gold-answer NLL** as the offline ground truth in the MiniGate.

## First run

Use BF16 for the first experiment.

```bash
python -m evaluation.gate0.run_poc \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --budget-curve-path evaluation/curve_data/llama-3.1-8b/snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy \
  --text-file /path/to/one_long_prompt.txt \
  --compression-ratio 0.80 \
  --suffix-len 64 \
  --swap-size 16 \
  --num-swaps 3 \
  --max-tokens 8192 \
  --dtype bfloat16 \
  --output results/gate0/poc_sample_001.json
```

The current `compression-ratio=0.80` means 80% pruning / roughly 20% retention, matching LU-KV's existing convention.

## Engineering sanity checks

Before interpreting any scientific result, verify:

1. FullKV compared with itself gives overlap approximately 1.
2. LU-KV baseline is deterministic across repeated runs.
3. Every candidate has the same total keep count as LU-KV.
4. A donor loses exactly `swap-size` entries and the receiver gains exactly the same amount.
5. No donor crosses the sink/recent-window minimum.
6. All candidates use the same SnapKV token scorer.

## Current limitation

The upstream LU-KV head-wise path stores the original K/V tensors and records pruned positions in `module.masked_key_indices`. During decoding, the attention patch replaces masked keys with fake keys so they receive approximately zero attention.

Therefore this branch is suitable for **Gate 0 signal validation**, but it does not yet establish that donor/receiver swaps move physical cache pages or reduce peak memory. That requires the separate Gate -1 paged/ragged-cache implementation audit.

## Next step after the 2-sample POC

If the engineering sanity checks pass:

- run 24 independent 8K prompts;
- use six swaps per prompt;
- evaluate 32- and 64-token probes;
- add LU marginal, attention, entropy, and local reconstruction proxies;
- use gold-answer NLL as the primary dense offline direction label;
- compute prompt-clustered sign accuracy and within-prompt Spearman correlation.

Stop if the behavior signal does not exceed the pre-registered MiniGate threshold or does not beat the strongest cheap proxy.
