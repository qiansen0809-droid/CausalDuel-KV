TASKS=(
    "4096"
)
RATIOS=(0.8)
TARGET_GPU=3
mkdir -p ./log

for ratio in "${RATIOS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Launching task: $task on GPU: $TARGET_GPU with Ratio: $ratio"
        python3 evaluate.py \
            --dataset ruler \
            --data_dir "$task" \
            --model /ssd1/models/llama-3.1-8b \
            --device "cuda:$TARGET_GPU" \
            --press_name lu_ea \
            --budget_curve_path  "curve_data/llama-3.1-8b/ea_0.02_sink4_win1_llama_avg_ratio.npy" \
            --compression_ratio "$ratio" > "./log/${task}_${ratio}.log" 2>&1

        echo "Task $task (Ratio $ratio) finished."
    done
done