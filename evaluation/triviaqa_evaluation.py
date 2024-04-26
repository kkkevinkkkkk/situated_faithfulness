# -*- coding: utf-8 -*-
""" Official evaluation script for v1.0 of the TriviaQA dataset.
Extended from the evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import sys
import argparse

import numpy as np

import utils.dataset_utils
import utils.utils


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def recall_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def exact_match_score_relax(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def is_exact_match(answer_object, prediction):
    ground_truths = get_ground_truths(answer_object)
    for ground_truth in ground_truths:
        if exact_match_score(prediction, ground_truth):
            return True
    return False


def has_exact_match(ground_truths, candidates):
    for ground_truth in ground_truths:
        if ground_truth in candidates:
            return True
    return False


def get_ground_truths(answer):
    return answer['NormalizedAliases'] + [normalize_answer(ans) for ans in answer.get('HumanAnswers', [])]


def get_oracle_score(ground_truth, predicted_answers, qid_list=None, mute=False):
    exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = normalize_answer(predicted_answers[qid])
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = has_exact_match(ground_truths, prediction)
        exact_match += int(em_for_this_question)

    exact_match = 100.0 * exact_match / len(qid_list)

    return {'oracle_exact_match': exact_match, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth)}

def evaluate_triviaqa_row(row):
    prediction = row['model_answer']
    ground_truths = get_ground_truths(row['Answer'])
    other_predictions = row['model_other_answers']
    predictions = [prediction] + other_predictions

    expected_em_relax = []
    for pred in predictions:
        expected_em_relax.append(metric_max_over_ground_truths(
            exact_match_score_relax, pred, ground_truths))
    expected_em_relax = np.mean(expected_em_relax)

    # em_for_this_question = metric_max_over_ground_truths(
    #     exact_match_score, prediction, ground_truths)
    # em_for_this_question_relax = metric_max_over_ground_truths(
    #     exact_match_score_relax, prediction, ground_truths)
    #
    # f1_for_this_question = metric_max_over_ground_truths(
    #     f1_score, prediction, ground_truths)
    #
    # recall_for_this_question = metric_max_over_ground_truths(
    #     recall_score, prediction, ground_truths)

    return expected_em_relax


def evaluate_triviaqa_df(df, mute=True):
    f1_scores = []
    exact_match_scores = []
    exact_match_scores_relax = []
    recall_scores = []
    scores = []

    for i, row in df.iterrows():
        prediction = row['model_answer']
        # prediction = row['generated_text']
        ground_truths = get_ground_truths(row['Answer'])

        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        em_for_this_question_relax = metric_max_over_ground_truths(
            exact_match_score_relax, prediction, ground_truths)

        if em_for_this_question == 0 and not mute:
            print("em=0:", prediction, ground_truths)

        exact_match_scores.append(em_for_this_question)
        exact_match_scores_relax.append(em_for_this_question_relax)

        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

        recall_for_this_question = metric_max_over_ground_truths(
            recall_score, prediction, ground_truths)
        f1_scores.append(f1_for_this_question)
        recall_scores.append(recall_for_this_question)
        scores.append({'f1': f1_for_this_question, 'em': em_for_this_question, 'recall': recall_for_this_question, 'em_relax': em_for_this_question_relax})


    f1 = 100.0 * np.mean(f1_scores)
    recall = 100.0 * np.mean(recall_scores)
    exact_match = 100.0 * np.mean(exact_match_scores)
    exact_match_relax = 100.0 * np.mean(exact_match_scores_relax)
    final_scores = {'exact_match': exact_match, 'f1': f1, 'denominator': len(df),
            'exact_match_relax': exact_match_relax, 'recall': recall}

    return final_scores, scores

def evaluate_triviaqa(ground_truth, predicted_answers, qid_list=None, mute=False):
    f1 = exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = 'Missed question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        if qid not in ground_truth:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = predicted_answers[qid]
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        if em_for_this_question == 0 and not mute:
            print("em=0:", prediction, ground_truths)
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question

    exact_match = 100.0 * exact_match / len(qid_list)
    f1 = 100.0 * f1 / len(qid_list)

    return {'exact_match': exact_match, 'f1': f1, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth)}


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation for TriviaQA {}'.format(expected_version))
    parser.add_argument('--dataset_file', help='Dataset file')
    parser.add_argument('--prediction_file', help='Prediction File')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    expected_version = 1.0
    args = get_args()

    dataset_json = utils.dataset_utils.read_triviaqa_data(args.dataset_file)
    if dataset_json['Version'] != expected_version:
        print('Evaluation expects v-{} , but got dataset with v-{}'.format(expected_version,dataset_json['Version']),
              file=sys.stderr)
    key_to_ground_truth = utils.dataset_utils.get_key_to_ground_truth(dataset_json)
    predictions = utils.utils.read_json(args.prediction_file)
    eval_dict = evaluate_triviaqa(key_to_ground_truth, predictions)
    print(eval_dict)
