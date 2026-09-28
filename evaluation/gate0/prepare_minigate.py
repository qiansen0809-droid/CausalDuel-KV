from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


RULER_RETRIEVAL_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
]

SINGLE_DOC_QUOTAS = {
    "narrativeqa": 3,
    "qasper": 3,
    "multifieldqa_en": 2,
}

MULTI_DOC_QUOTAS = {
    "hotpotqa": 3,
    "2wikimqa": 3,
    "musique": 2,
}


def parse_args():
    p = argparse.ArgumentParser(description="Prepare the fixed 24-prompt Gate-0 MiniGate set.")
    p.add_argument("--model", required=True, help="Tokenizer path used for exact token counts.")
    p.add_argument("--output", type=Path, default=Path("results/gate0/minigate/minigate_24.jsonl"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/gate0/minigate/minigate_24_manifest.json"),
    )
    p.add_argument("--target-context-tokens", type=int, default=8192)
    p.add_argument("--min-context-tokens", type=int, default=6144)
    p.add_argument("--max-context-tokens", type=int, default=9216)
    p.add_argument("--ruler-revision", default="24adcea")
    p.add_argument(
        "--longbench-revision",
        default="0ce23c4aa955accf17527097eb12a8f00e2743e6",
    )
    return p.parse_args()


def as_answers(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def token_len(tokenizer, text, add_special_tokens):
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def make_record(
    *,
    family,
    task,
    source_dataset,
    source_config,
    source_revision,
    source_index,
    row,
    tokenizer,
):
    context = str(row["context"])
    question = str(row.get("question", ""))
    answer_prefix = str(row.get("answer_prefix", ""))
    answers = as_answers(row.get("answer", row.get("answers")))

    if not answers:
        raise ValueError(f"{family}/{task}/{source_index} has no gold answer")

    context_tokens = token_len(tokenizer, context, add_special_tokens=True)
    query_tokens = token_len(tokenizer, question + answer_prefix, add_special_tokens=False)

    return {
        "id": f"{family}__{task}__{source_index}",
        "family": family,
        "task": task,
        "source_dataset": source_dataset,
        "source_config": source_config,
        "source_revision": source_revision,
        "source_index": int(source_index),
        "context": context,
        "question": question,
        "answer_prefix": answer_prefix,
        "answers": answers,
        "max_new_tokens": int(row.get("max_new_tokens", 64)),
        "context_tokens": int(context_tokens),
        "query_tokens": int(query_tokens),
        "prompt_tokens": int(context_tokens + query_tokens),
    }


def choose_by_length(records, quota, target, low, high, label):
    eligible = [
        r for r in records
        if low <= r["context_tokens"] <= high
    ]
    if len(eligible) < quota:
        raise RuntimeError(
            f"{label}: only {len(eligible)} examples fall in "
            f"[{low}, {high}] context tokens; need {quota}. "
            "Do not silently truncate benchmark contexts; widen the preregistered "
            "length window explicitly if this happens."
        )

    eligible.sort(
        key=lambda r: (
            abs(r["context_tokens"] - target),
            r["source_index"],
        )
    )
    return eligible[:quota]


def load_longbench_group(
    tokenizer,
    quotas,
    family,
    revision,
    target,
    low,
    high,
):
    selected = []
    for task, quota in quotas.items():
        ds = load_dataset(
            "Xnhyacinth/LongBench",
            task,
            split="test",
            revision=revision,
        )

        records = [
            make_record(
                family=family,
                task=task,
                source_dataset="Xnhyacinth/LongBench",
                source_config=task,
                source_revision=revision,
                source_index=i,
                row=row,
                tokenizer=tokenizer,
            )
            for i, row in enumerate(ds)
        ]
        selected.extend(
            choose_by_length(
                records,
                quota=quota,
                target=target,
                low=low,
                high=high,
                label=f"{family}/{task}",
            )
        )
    return selected


def load_ruler(tokenizer, revision, target):
    ds = load_dataset(
        "simonjegou/ruler",
        "8192",
        split="test",
        revision=revision,
    )

    by_task = {task: [] for task in RULER_RETRIEVAL_TASKS}
    for i, row in enumerate(ds):
        task = str(row["task"])
        if task not in by_task:
            continue
        by_task[task].append(
            make_record(
                family="ruler_retrieval",
                task=task,
                source_dataset="simonjegou/ruler",
                source_config="8192",
                source_revision=revision,
                source_index=i,
                row=row,
                tokenizer=tokenizer,
            )
        )

    selected = []
    for task in RULER_RETRIEVAL_TASKS:
        if not by_task[task]:
            raise RuntimeError(f"RULER config 8192 has no rows for task={task}")

        # One prompt per official retrieval subtask. Length is the only ranking
        # criterion; no model outputs or gold-answer quality are consulted.
        candidates = sorted(
            by_task[task],
            key=lambda r: (
                abs(r["context_tokens"] - target),
                r["source_index"],
            ),
        )
        selected.append(candidates[0])

    return selected


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ruler = load_ruler(
        tokenizer,
        revision=args.ruler_revision,
        target=args.target_context_tokens,
    )
    single = load_longbench_group(
        tokenizer,
        quotas=SINGLE_DOC_QUOTAS,
        family="longbench_single",
        revision=args.longbench_revision,
        target=args.target_context_tokens,
        low=args.min_context_tokens,
        high=args.max_context_tokens,
    )
    multi = load_longbench_group(
        tokenizer,
        quotas=MULTI_DOC_QUOTAS,
        family="longbench_multi",
        revision=args.longbench_revision,
        target=args.target_context_tokens,
        low=args.min_context_tokens,
        high=args.max_context_tokens,
    )

    records = ruler + single + multi
    counts = {}
    for row in records:
        counts[row["family"]] = counts.get(row["family"], 0) + 1

    expected = {
        "ruler_retrieval": 8,
        "longbench_single": 8,
        "longbench_multi": 8,
    }
    if counts != expected:
        raise AssertionError(f"MiniGate family counts mismatch: {counts} != {expected}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "protocol": "CausalDuel-KV Gate-0 MiniGate",
        "num_prompts": len(records),
        "family_counts": counts,
        "target_context_tokens": args.target_context_tokens,
        "longbench_context_token_window": [
            args.min_context_tokens,
            args.max_context_tokens,
        ],
        "ruler": {
            "dataset": "simonjegou/ruler",
            "config": "8192",
            "revision": args.ruler_revision,
            "tasks": RULER_RETRIEVAL_TASKS,
            "selection": "one row per retrieval subtask, closest context length to target",
        },
        "longbench": {
            "dataset": "Xnhyacinth/LongBench",
            "revision": args.longbench_revision,
            "single_doc_quotas": SINGLE_DOC_QUOTAS,
            "multi_doc_quotas": MULTI_DOC_QUOTAS,
            "selection": "closest natural context lengths to target within fixed token window; no truncation",
        },
        "samples": [
            {
                key: row[key]
                for key in (
                    "id",
                    "family",
                    "task",
                    "source_dataset",
                    "source_config",
                    "source_revision",
                    "source_index",
                    "context_tokens",
                    "query_tokens",
                    "prompt_tokens",
                    "answers",
                    "max_new_tokens",
                )
            }
            for row in records
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote MiniGate data: {args.output}")
    print(f"Wrote manifest:      {args.manifest}")
    print("Family counts:", counts)
    for row in records:
        print(
            row["id"],
            f"context={row['context_tokens']}",
            f"query={row['query_tokens']}",
            f"prompt={row['prompt_tokens']}",
        )


if __name__ == "__main__":
    main()
