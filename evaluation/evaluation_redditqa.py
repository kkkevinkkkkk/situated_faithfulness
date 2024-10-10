from .evaluation_triviaqa import normalize_answer
import numpy as np
import re

def get_answer(x):
    # x = normalize_answer(x)
    original_answer = x
    # extract A), B), C), D) from the answer
    x = x.strip()
    # x = x.split(") ")[0]
    pattern = r'\b[A-D]\)'
    pattern = r'([A-D])\)'
    matches = re.findall(pattern, x)
    if len(matches) > 0:
        x = matches[-1]
    else:
        x = x.split(":")[-1].strip()
        if len(x) != 1:
            print("wrong extraction for ", original_answer)
    return x



def evaluate_redditqa(df):

    df["final_pred"] = df["generated_text"].apply(lambda x: get_answer(x))
    df["normed_answer"] = df["answer"].apply(lambda x: get_answer(x))

    scores = df["final_pred"] == df["normed_answer"]
    total_scores = {"accuracy": scores.mean()}
    return total_scores, scores

def evaluate_redditqa_row(row):
    ground_truth = get_answer(row["answer"])
    model_answer0 = row["generated_text"]
    other_answers = row["other_answers"]
    model_answers = [model_answer0] + other_answers
    scores = []
    for model_answer in model_answers:
        model_answer_ = get_answer(model_answer)
        scores.append(model_answer_ == ground_truth)

    row["expected_correctness"] = np.mean(scores)
    return row
