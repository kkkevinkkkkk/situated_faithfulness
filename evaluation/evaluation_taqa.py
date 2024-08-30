from evaluation import f1_score_token_level, exact_match_score_relax, recall_score_token_level
from evaluation import metric_max_over_ground_truths
import numpy as np
scores = []

def evaluate_taqa_row(row):
    row['target_answer'] = row['answer']['2022']
    recall = metric_max_over_ground_truths(recall_score_token_level, row['generated_text'], row['target_answer'])
    f1 = metric_max_over_ground_truths(f1_score_token_level, row['generated_text'], row['target_answer'])
    exact_match = metric_max_over_ground_truths(exact_match_score_relax, row['generated_text'], row['target_answer'])
    row["scores"] = {
        'recall': recall,
        'f1': f1,
        'exact_match_score_relax': exact_match
    }
    return row
def evaluate_taqa_expected_correctness(row):
    model_answers = [row['generated_text']] + row['other_answers']
    row_copy = row.copy()
    f1_scores = []
    for model_answer in model_answers:
        row_copy['generated_text'] = model_answer
        f1_score = evaluate_taqa_row(row_copy)["scores"]["f1"]
        f1_scores.append(f1_score)
    row["expected_correctness"] = np.mean(f1_scores)
    return row

def evaluate_taqa_df(df, return_df=False):
    df = df.apply(lambda x: evaluate_taqa_row(x), axis=1)
    scores = df["scores"]
    final_f1 = df["scores"].apply(lambda x: x["f1"]).mean()
    final_recall = df["scores"].apply(lambda x: x["recall"]).mean()
    final_exact_match = df["scores"].apply(lambda x: x["exact_match_score_relax"]).mean()
    final_scores = {
        'final_f1': final_f1,
        'final_recall': final_recall,
        'final_exact_match': final_exact_match
    }
    df["final_scores"] = [final_scores] * df.shape[0]
    if return_df:
        return df
    return final_scores, scores