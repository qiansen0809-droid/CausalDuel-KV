#!/bin/bash
set -euo pipefail

# ==============================================================
# Configuration
# ==============================================================

# Short name used for directory organisation (no slashes)
MODEL_NAME="llama-3.1-8b"

# Full path to the model weights
MODEL_PATH="/ssd1/models/Llama-3.1-8B-Instruct"

# Press / method name (must match PRESS_REGISTRY in evaluate_registry.py)
PRESS_NAME="lu_ea"

# Path to the pre-computed budget curve (.npy); leave empty "" if not needed
BUDGET_CURVE_PATH="curve_data/llama-3.1-8b/ea_0.02_sink4_win1_llama_avg_ratio.npy"

# LongBench tasks to evaluate (16 standard tasks)
TASKS=(
    "narrativeqa"
    "qasper"
    "multifieldqa_en"
    "hotpotqa"
    "2wikimqa"
    "musique"
    "gov_report"
    "qmsum"
    "multi_news"
    "trec"
    "triviaqa"
    "samsum"
    "passage_count"
    "passage_retrieval_en"
    "lcc"
    "repobench-p"
)

# Compression ratios to sweep
RATIOS=(0.8 0.9)

# GPU index
TARGET_GPU=2

# Root directory for all results (relative to this script's location)
RESULTS_ROOT="results"

# ==============================================================
# Execution
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for ratio in "${RATIOS[@]}"; do

    OUT_DIR="${SCRIPT_DIR}/${RESULTS_ROOT}/${MODEL_NAME}/${PRESS_NAME}/${ratio}"
    mkdir -p "${OUT_DIR}"

    for task in "${TASKS[@]}"; do

        LOG_FILE="${OUT_DIR}/${task}.log"

        echo "=================================================="
        echo "  Model  : ${MODEL_NAME}"
        echo "  Press  : ${PRESS_NAME}"
        echo "  Task   : ${task}"
        echo "  Ratio  : ${ratio}"
        echo "  Output : ${OUT_DIR}"
        echo "  Log    : ${LOG_FILE}"
        echo "=================================================="

        EXTRA_ARGS=()
        if [ -n "${BUDGET_CURVE_PATH}" ]; then
            EXTRA_ARGS+=(--budget_curve_path "${BUDGET_CURVE_PATH}")
        fi

        python3 "${SCRIPT_DIR}/evaluate.py" \
            --dataset longbench \
            --data_dir "${task}" \
            --model "${MODEL_PATH}" \
            --device "cuda:${TARGET_GPU}" \
            --press_name "${PRESS_NAME}" \
            --compression_ratio "${ratio}" \
            --output_dir "${OUT_DIR}" \
            "${EXTRA_ARGS[@]}" \
            > "${LOG_FILE}" 2>&1

        echo "Done: ${MODEL_NAME}/${PRESS_NAME}/${ratio}/${task}"
    done

    echo "=================================================="
    echo "  All tasks finished for ratio ${ratio}"
    echo "=================================================="
done
