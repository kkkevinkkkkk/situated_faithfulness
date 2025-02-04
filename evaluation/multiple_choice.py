import re
from .evaluation_triviaqa import recall_score as recall_score_token_level
import numpy as np

def extract_answer_letter(datapoint):
    question = datapoint["question"]
    answer = datapoint["generated_text"]
    pattern = r'\b[A-Z]\)|\([A-Z]\)'
    # Find all matches in the string
    matches = re.findall(pattern, answer)
    # Extract just the letters from the matches, removing any surrounding parentheses
    letters = [match.strip('()') for match in matches]
    return letters[0] if len(letters) > 0 else None


def extract_answer(datapoint):
    question = datapoint["question"]
    generated_text = datapoint["generated_text"]
    answer = datapoint["answer"]
    candidates = datapoint["candidates_to_letters"].values()
    recall_scores = [recall_score_token_level(generated_text, candidate) for candidate in candidates]
    answers_max_recall = [candidate for candidate, recall_score in zip(candidates, recall_scores) if
                          recall_score == max(recall_scores)]
    model_answer = answers_max_recall[0]
    if len(answers_max_recall) > 1:
        letter = extract_answer_letter(datapoint)
        if letter is None or letter not in datapoint["candidates_to_letters"].keys():
            print("No match found for the answer:", generated_text)
            return ""
        model_answer = datapoint["candidates_to_letters"][letter]
        print(f"Question: {question}")
        print(f" Answer: {answer}, Generated Text: {generated_text}")
        print(answers_max_recall)

    return model_answer


def evaluate_multiple_choice(df):
    scores = df.apply(lambda x: {"exact_match": x["answer_letter"] == x["output"],
                                 "em_relax": x["answer_letter"] == x["output"]}, axis=1)
    exact_match = np.mean([score["exact_match"] for score in scores])
    final_scores = {"exact_match": exact_match}
    return final_scores, scores