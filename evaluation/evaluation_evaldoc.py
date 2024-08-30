from .evaluation_triviaqa import normalize_answer


def get_answer(x):
    x = normalize_answer(x)
    wrong_words = ["wrong", "incorrect", "error"]
    correct_words = ["correct", "accurate"]

    def check_word(x, words):
        for word in words:
            if word in x:
                return True
        return False

    if check_word(x, correct_words) and not check_word(x, wrong_words):
        return 1
    elif check_word(x, wrong_words):
        return 0
    else:
        return -1


def get_answer2(x):
    x = x.strip().lower()

    if "correct" in x and "incorrect" not in x:
        return 1
    elif "wrong" in x or "incorrect" in x:
        return 0
    else:
        return 0
def evaluate_evaldoc(df):

    df["final_pred"] = df["generated_text"].apply(lambda x: get_answer(x))
    scores = df["final_pred"] == df["label"]
    total_scores = {"accuracy": scores.mean()}
    return total_scores, scores

def evaluate_evaldoc_expected_correctness(row):
    ground_truth = row["label"]
    model_answer = row["generated_text"]
    other_answers = row["other_answers"]
    model_answers = [model_answer] + other_answers
    scores = []
    for model_answer in model_answers:
        model_answer_ = get_answer(model_answer)
        scores.append(model_answer_ == ground_truth)
    row["expected_correctness"] = np.mean(scores)

    return row