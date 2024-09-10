import numpy as np
def evaluate_selfeval(df):
    df["final_pred"] = df["generated_text"].apply(
        lambda x: True if "true" in x.lower().strip() or "correct" in x.lower().strip() else False)
    # df["label"] = df["label"].apply(lambda x: True if x == "True" else False)
    scores = df["final_pred"] == df["label"]
    total_scores = {"accuracy": scores.mean()}
    return total_scores, scores

def evaluate_selfeval_expected_correctness(row):
    ground_truth = row["label"]
    model_answer = row["generated_text"]
    other_answers = row["other_answers"]
    model_answers = [model_answer] + other_answers
    scores = []
    for model_answer in model_answers:
        model_answer_ = True if "true" in model_answer.lower().strip() or "correct" in model_answer.lower().strip() else False
        scores.append(model_answer_ == ground_truth)
    row["expected_correctness"] = np.mean(scores)

    return row

