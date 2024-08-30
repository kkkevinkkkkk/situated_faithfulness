from .utils import (normalize_answer, metric_max_over_ground_truths,
                    exact_match_score, exact_match_score_relax, f1_score, recall_score)

import numpy as np
def evaluate_conflictqa(df):
    full_metrics_scores = []
    scores = []
    for i, row in df.iterrows():
        metrics = evaluate_conflictqa_single_answer(row)
        full_metrics_scores.append(metrics)
        scores.append(metrics["em_relax"])

    df["scores"] = full_metrics_scores
    df["score"] = scores
    recall = df["scores"].apply(lambda x: x["recall"]).mean()
    em = df["scores"].apply(lambda x: x["em"]).mean()
    em_relax = df["scores"].apply(lambda x: x["em_relax"]).mean()
    f1 = df["scores"].apply(lambda x: x["f1"]).mean()
    total_scores = {"f1": f1, "em": em, "recall": recall, "em_relax": em_relax}
    return total_scores, scores


def evaluate_conflictqa_single_answer(row):
    prediction = row['generated_text']
    ground_truths = row["ground_truth"]
    em_for_this_question = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths)
    em_for_this_question_relax = metric_max_over_ground_truths(
        exact_match_score_relax, prediction, ground_truths)

    f1_for_this_question = metric_max_over_ground_truths(
        f1_score, prediction, ground_truths)
    recall_for_this_question = metric_max_over_ground_truths(
        recall_score, prediction, ground_truths)
    scores = {'f1': f1_for_this_question, 'em': em_for_this_question, 'recall': recall_for_this_question, 'em_relax': em_for_this_question_relax}
    return scores

def evaluate_conflictqa_row(row):
    model_answer0 = row['generated_text']
    # ground_truth = row["ground_truth"]
    other_answers = row["other_answers"]
    model_answers = [model_answer0] + other_answers
    scores = []
    for model_answer in model_answers:
        model_answer_ = model_answer
        row["generated_text"] = model_answer_
        scores.append(evaluate_conflictqa_single_answer(row)["em_relax"])

    row["generated_text"] = model_answer0
    row["expected_correctness"] = np.mean(scores)
    return row

