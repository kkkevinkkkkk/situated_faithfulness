from .utils import (normalize_answer, metric_max_over_ground_truths,
                    exact_match_score, exact_match_score_relax, f1_score, recall_score)

import numpy as np
from .evaluation_naturalqa import evaluate_naturalqa_answer_gpt4 as evaluate_answer_gpt4
from utils import multi_process_map
# USE_GPT4 = True
USE_GPT4 = False
def evaluate_clasheval(df):
    full_metrics_scores = []
    scores = []


    results = multi_process_map(df,evaluate_clasheval_single_answer,  num_proc=64)
    # df["scores"] = results
    df["score"] = results.apply(lambda x: x["em_relax"] if not USE_GPT4 else x["gpt_score"], axis=1)
    scores = df["score"].tolist()
    # df["scores"] = full_metrics_scores
    # df["score"] = scores
    # recall = results.apply(lambda x: x["recall"]).mean()
    # em = results.apply(lambda x: x["em"]).mean()
    em_relax = results.apply(lambda x: x["em_relax"], axis=1).mean()
    # f1 = results.apply(lambda x: x["f1"]).mean()
    # total_scores = {"f1": f1, "em": em, "recall": recall, "em_relax": em_relax}
    total_scores = {"em_relax": em_relax}
    return total_scores, scores


def evaluate_clasheval_single_answer(row):
    prediction = row['generated_text']
    # ground_truths = [row['answer']]
    ground_truths = row["answers"]
    # em_for_this_question = metric_max_over_ground_truths(
        # exact_match_score, prediction, ground_truths)
    em_for_this_question_relax = metric_max_over_ground_truths(
        exact_match_score_relax, prediction, ground_truths)

    # f1_for_this_question = metric_max_over_ground_truths(
    #     f1_score, prediction, ground_truths)
    # recall_for_this_question = metric_max_over_ground_truths(
    #     recall_score, prediction, ground_truths)

    gpt_score = evaluate_answer_gpt4(row) if USE_GPT4 else 0

    scores = {"em_relax": em_for_this_question_relax, "gpt_score": gpt_score}

    # scores = {'f1': f1_for_this_question, 'em': em_for_this_question, 'recall': recall_for_this_question, 'em_relax': em_for_this_question_relax, "gpt_score": gpt_score}
    return scores

def evaluate_clasheval_row(row):
    model_answer0 = row['generated_text']
    # ground_truth = row["ground_truth"]
    other_answers = row["other_answers"]
    model_answers = [model_answer0] + other_answers
    scores = []
    for model_answer in model_answers:
        model_answer_ = model_answer
        row["generated_text"] = model_answer_
        score_key = "em_relax" if not USE_GPT4 else "gpt_score"
        scores.append(evaluate_clasheval_single_answer(row)[score_key])

    row["generated_text"] = model_answer0
    row["expected_correctness"] = np.mean(scores)
    return row

