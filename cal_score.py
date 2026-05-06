import pandas as pd
import json
import os
import re
import string
import numpy as np
from collections import Counter
from rouge import Rouge

# Optional imports; not required for gov_report (which only uses rouge)
try:
    import jieba
    from fuzzywuzzy import fuzz
except ImportError:
    pass

# ==========================================
# Metric computation code (inlined)
# ==========================================

def calculate_metrics(df):
    predictions = df["predicted_answer"].tolist()
    answers = df["answers"].tolist()
    # Assumes all rows in df belong to the same task
    dataset = df["task"].tolist()[0] 
    
    # Fall back to None if the all_classes column is missing
    if "all_classes" in df.columns:
        all_classes = df["all_classes"].tolist()[0]
    else:
        all_classes = None

    # Compute the score as a float
    score_value = scorer(dataset, predictions, answers, all_classes)
    
    return {dataset: score_value}

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_zh_answer(s):
    def white_space_fix(text):
        return "".join(text.split())
    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception as e:
        # print(f"An error occurred: {e}")
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# ==========================================
# Main execution logic
# ==========================================

def main():
    # Path to the predictions CSV file
    csv_path = "/ssd2/tangziyao/kvpress-0.2.10/scripts/llama_3.1_8b/longbench_v1/fast/attn_max_sink_results/longbench__gov_report__--ssd2--tangziyao--tzy--models--llama-3.1-8b__mykeydiff__0.80/predictions.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Original Columns: {df.columns.tolist()}")

    # 1. Filter to only include gov_report rows
    # (usually a no-op if the file is from the gov_report directory, but kept for safety)
    target_task = "gov_report"
    if "task" in df.columns:
        # Use fuzzy match in case the task name varies (e.g. "gov_report" vs "LongBench/gov_report")
        df = df[df['task'].astype(str).str.contains(target_task, case=False, na=False)].copy()
        print(f"Filtered to '{target_task}' task. Rows: {len(df)}")
        # Normalize the task name so dataset2metric can find the rouge_score entry
        df['task'] = target_task
    else:
        # No task column — assume gov_report
        print(f"Warning: 'task' column missing. Assuming '{target_task}'.")
        df['task'] = target_task

    if len(df) == 0:
        print("No data found for this task.")
        return

    # 2. Convert 'answers' from its stringified list form back to an actual Python list
    # Also handle NaN values in predicted_answer
    df["predicted_answer"] = df["predicted_answer"].fillna("").astype(str)
    
    print("Converting 'answers' from string to list...")
    try:
        # Use eval to parse the stringified list "['answer']" back to ['answer']
        df["answers"] = df["answers"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    except Exception as e:
        print(f"Error converting answers column: {e}")
        return

    # Parse all_classes if present
    if "all_classes" in df.columns:
        df["all_classes"] = df["all_classes"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    else:
        df["all_classes"] = None

    # 3. Compute metrics
    print("Calculating metrics...")
    try:
        results = calculate_metrics(df)
        print("\n" + "="*40)
        print("Final Results:")
        print(json.dumps(results, indent=4))
        print("="*40)
        
        # Save results to file
        output_file = csv_path.replace(".csv", "_manual_metrics.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Metrics saved to: {output_file}")
        
    except Exception as e:
        print(f"Calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()