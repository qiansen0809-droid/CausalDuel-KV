import numpy as np
import heapq
import os
import glob
import argparse
import multiprocessing as mp
from tqdm import tqdm

def compute_lower_convex_hull(y_values):
    n = len(y_values)
    stack = [0]
    for i in range(1, n):
        while len(stack) >= 2:
            p1, p2, p3 = stack[-2], stack[-1], i
            x1, y1 = p1, y_values[p1]
            x2, y2 = p2, y_values[p2]
            x3, y3 = p3, y_values[p3]
            cross_product = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if cross_product <= 0:
                stack.pop()
            else:
                break
        stack.append(i)
    return stack

def apply_convex_hull_smoothing(aligned_gt):
    rows, cols, length = aligned_gt.shape
    smoothed_gt = np.zeros_like(aligned_gt)
    for r in range(rows):
        for c in range(cols):
            raw_vals = aligned_gt[r, c]
            vals_to_discard = raw_vals[::-1] 
            cum_loss = np.concatenate(([0], np.cumsum(vals_to_discard)))
            hull_indices = compute_lower_convex_hull(cum_loss)
            reconstructed_vals = np.zeros(length)
            for i in range(len(hull_indices) - 1):
                idx_start, idx_end = hull_indices[i], hull_indices[i+1]
                slope = (cum_loss[idx_end] - cum_loss[idx_start]) / (idx_end - idx_start)
                reconstructed_vals[idx_start : idx_end] = slope
            smoothed_gt[r, c] = reconstructed_vals[::-1]
    return smoothed_gt

def compute_optimal_budget_for_pair(data_gt, data_pred, sink_size, window_size, prune_threshold_int=99, layerwise=True):
    """Returns the 99-step curve results array for a single question pairing, straight from memory"""
    limit_ratio = prune_threshold_int / 100.0
    rows, cols, length = data_gt.shape
    eff_len = length - sink_size - window_size
    
    if eff_len <= 0: return None

    mid_gt = data_gt[:, :, sink_size:length-window_size]
    mid_pred = data_pred[:, :, sink_size:length-window_size]
    aligned_mid_gt = np.zeros_like(mid_gt)
    
    for r in range(rows):
        for c in range(cols):
            sort_indices = np.argsort(mid_pred[r, c])[::-1]
            aligned_mid_gt[r, c] = mid_gt[r, c][sort_indices]

    priority_scores_mid = apply_convex_hull_smoothing(aligned_mid_gt)

    results = np.zeros((99, rows, cols), dtype=np.float64)
    user_min_keep = int(round((1.0 - limit_ratio) * length))
    min_keep_limit = max(user_min_keep, sink_size + window_size)

    if layerwise:
        total_elements = rows * cols * length
        current_keep_count = np.full((rows, cols), length, dtype=np.int32)
        total_discarded = 0
        heap = []
        for r in range(rows):
            for c in range(cols):
                if eff_len - 1 >= 0:
                    val = priority_scores_mid[r, c, eff_len - 1]
                    heapq.heappush(heap, (val, r, c, eff_len - 1))

        for i in range(99):
            target_d = int(round(((i + 1) / 100.0) * total_elements))
            global_x = (i + 1) / 100.0
            if global_x > limit_ratio:
                results[i] = np.full((rows, cols), global_x)
                continue

            while total_discarded < target_d and heap:
                val, r, c, idx_in_mid = heapq.heappop(heap)
                if current_keep_count[r, c] > min_keep_limit:
                    current_keep_count[r, c] -= 1
                    total_discarded += 1
                    next_idx = idx_in_mid - 1
                    if next_idx >= 0:
                        heapq.heappush(heap, (priority_scores_mid[r, c, next_idx], r, c, next_idx))
            results[i] = 1.0 - (current_keep_count / length)
    else:
        layer_elements = cols * length
        for r in range(rows):
            current_keep_count_layer = np.full(cols, length, dtype=np.int32)
            total_discarded_layer = 0
            heap = []
            for c in range(cols):
                if eff_len - 1 >= 0:
                    val = priority_scores_mid[r, c, eff_len - 1]
                    heapq.heappush(heap, (val, c, eff_len - 1))

            for i in range(99):
                target_d_layer = int(round(((i + 1) / 100.0) * layer_elements))
                global_x = (i + 1) / 100.0
                if global_x > limit_ratio:
                    results[i, r, :] = global_x
                    continue

                while total_discarded_layer < target_d_layer and heap:
                    val, c, idx_in_mid = heapq.heappop(heap)
                    if current_keep_count_layer[c] > min_keep_limit:
                        current_keep_count_layer[c] -= 1
                        total_discarded_layer += 1
                        next_idx = idx_in_mid - 1
                        if next_idx >= 0:
                            heapq.heappush(heap, (priority_scores_mid[r, c, next_idx], c, next_idx))
                results[i, r, :] = 1.0 - (current_keep_count_layer / length)

    return results

# ================= Multiprocessing Worker =================
def worker_task(task_args):
    """
    Worker function to load data and process a single question file.
    Must be at the top level to be picklable by multiprocessing.Pool.
    """
    q_file, pred_path, sink_size, window_size, threshold, layerwise = task_args
    try:
        data_gt = np.load(q_file)
        data_pred = np.load(pred_path)
        
        res = compute_optimal_budget_for_pair(
            data_gt, data_pred, 
            sink_size=sink_size, 
            window_size=window_size, 
            prune_threshold_int=threshold,
            layerwise=layerwise
        )
        return res
    except Exception as e:
        print(f"\n[Worker Error] Failed to process {q_file}: {e}")
        return None

# ================= Main Execution =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to step 1 temporary folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store the final curve files")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix naming for the output files")
    parser.add_argument("--configs", nargs='+', required=True, help="Method configs formatted as method:sink:window (e.g. ea:4:1 snapkv:4:32)")
    parser.add_argument("--threshold", type=int, default=99)
    parser.add_argument("--layerwise", action="store_true", help="Enable global budget allocation across layers")
    parser.add_argument("--num_workers", type=int, default=30, help="Number of concurrent processes")
    args = parser.parse_args()

    # Parse configs into a dictionary
    method_configs = {}
    for cfg in args.configs:
        parts = cfg.split(':')
        if len(parts) == 3:
            method_configs[parts[0]] = {'sink': int(parts[1]), 'window': int(parts[2])}
        else:
            print(f"Warning: Configuration '{cfg}' is invalid. Must be format 'method:sink:window'.")

    context_dirs = glob.glob(os.path.join(args.input_dir, "context_*"))
    if not context_dirs:
        print(f"Error: No context_* directories found in {args.input_dir}!")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    mode_str = "Global-Layerwise" if args.layerwise else "Layer-Independent"
    print(f"Starting Convex Hull & Averaging | Mode: {mode_str} | Workers: {args.num_workers}")
    
    # Process each method independently
    for method, config in method_configs.items():
        sink_size = config['sink']
        window_size = config['window']
        
        # 1. Gather all tasks for the current method
        tasks = []
        for c_dir in context_dirs:
            pred_path = os.path.join(c_dir, f"{method}.npy")
            if not os.path.exists(pred_path):
                continue
                
            q_files = glob.glob(os.path.join(c_dir, "question_*.npy"))
            for q_file in q_files:
                tasks.append((q_file, pred_path, sink_size, window_size, args.threshold, args.layerwise))

        total_tasks = len(tasks)
        if total_tasks == 0:
            print(f"[{method}] Skipped: No valid data/questions found.")
            continue
            
        total_sum = None
        valid_count = 0
        
        # 2. Execute tasks using multiprocessing Pool
        desc_str = f"Processing [{method}] (sink={sink_size}, win={window_size})"
        
        with mp.Pool(processes=args.num_workers) as pool:
            # imap_unordered provides a generator that yields as soon as a process finishes,
            # which is perfect for smooth tqdm updates.
            for res in tqdm(pool.imap_unordered(worker_task, tasks), total=total_tasks, desc=desc_str, unit="q"):
                if res is not None:
                    if total_sum is None:
                        total_sum = np.zeros_like(res, dtype=np.float64)
                    total_sum += res
                    valid_count += 1

        # 3. Save results for this method
        if valid_count > 0:
            avg_data = total_sum / valid_count
            out_path = os.path.join(args.output_dir, f"{args.output_prefix}_{method}_sink{sink_size}_win{window_size}.npy")
            np.save(out_path, avg_data)
            print(f"[{method}] Completed: Merged {valid_count} samples. Saved to: {out_path}\n")
        else:
            print(f"[{method}] Failed: All tasks returned None.\n")