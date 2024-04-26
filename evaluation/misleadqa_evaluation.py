from .triviaqa_evaluation import normalize_answer
import numpy as np
def get_answer(x):
    x = normalize_answer(x)
    if "yes" in x:
        return "yes"
    elif "no" in x:
        return "no"
    else:
        return "unknown"


def evaluate_misleadqa_fc(df):

    df["final_pred"] = df["generated_text"].apply(lambda x: get_answer(x))
    scores = df["final_pred"] == df["answer"]
    total_scores = {"accuracy": scores.mean()}
    return total_scores, scores

def evaluate_misleadqa_fc_row(row):
    ground_truth = row["answer"]
    model_answer = row["generated_text"]
    other_answers = row["other_answers"]
    model_answers = [model_answer] + other_answers
    scores = []
    for model_answer in model_answers:
        model_answer_ = get_answer(model_answer)
        scores.append(model_answer_ == ground_truth)

    return np.mean(scores)
