from utils import read_jsonl, retriever, save_jsonl, NERModel, TEMPLATES
from evaluation import evaluate_triviaqa_df, normalize_answer, f1_score_token_level, extract_answer
from evaluation import (evaluate_triviaqa_row, evaluate_multiple_choice, evaluate_evaldoc, get_answer_evaldoc,
                        evaluate_misleadqa_fc, get_answer_misleadqa_fc, evaluate_misleadqa_fc_row)

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from pipeline import pipeline_init, MyPipeline
import torch

import fire
import numpy as np
from copy import deepcopy
import re
import joblib
class Evaluator:
    def __init__(self, prediction_file="", df=None, recalibration_model_path=None):
        self.prediction_file = prediction_file
        if df is not None:
            self.predictions_df = df
        else:
            self.predictions_df = read_jsonl(self.prediction_file)
        self.retriever = None
        self.ner = None
        self.extractor = None
        self.multiple_choice = True if "multiple_choice" in self.prediction_file else False
        self.chain_of_confidence = True if "chain_of_confidence" in self.prediction_file else False
        self.dataset_name = None
        if "triviaqa" in self.prediction_file:
            self.dataset_name = "triviaqa"
        elif "evaldoc" in self.prediction_file:
            self.dataset_name = "evaldoc"
        elif "misleadqa_fc" in self.prediction_file:
            self.dataset_name = "misleadqa_fc"
        else:
            raise NotImplementedError()
        self.recalibration_model_path = recalibration_model_path
        self.recalibration_model = joblib.load(self.recalibration_model_path) if self.recalibration_model_path is not None else None

    def get_model_answer(self, eval_item, norm_method="", verbose=False):
        # if self.retriever is None:
        #     self.retriever = retriever.Retriever()
        # question = eval_item["question"]
        # generated_text = eval_item["generated_text"]
        # model_answer = self.retriever.retrieve_answer(answer=generated_text, question=question)

        if norm_method == "ner":
            if self.ner is None:
                self.ner = NERModel()
            entities = self.ner(eval_item["qa_statement"])
            groundtruth_entities = []
            for entity in entities:
                if normalize_answer(eval_item["answer"]) in normalize_answer(entity["word"]):
                    groundtruth_entities.append(entity)


            if len(groundtruth_entities) == 0:
                if verbose:
                    print("Warning: no entity detected")
                    print(eval_item["qa_statement"], eval_item["answer"], entities)
                    print()
                return eval_item["generated_text"]

            if len(groundtruth_entities) > 1 and verbose:
                print("Warning: multiple entities detected")

            entities = self.ner(eval_item["generated_text"])
            answers = [entity['word'] for entity in entities if
                       entity["entity_group"] == groundtruth_entities[0]["entity_group"]]

            model_answer = answers[0] if len(answers) > 0 else eval_item["generated_text"]
            for answer in answers:
                if normalize_answer(answer) in normalize_answer(eval_item["answer"]):
                    model_answer = answer
                    break

        elif norm_method == "extractor":
            if self.extractor is None:
                model_path = "/usr/xtmp/yh386/faithfulness/results/ft_extract_answer/3.0.0/checkpoint-final"
                model = AutoPeftModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                                 device_map="auto")
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
                self.extractor = pipeline_init(
                    task="text-generation",
                    model=model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    pipeline_class=MyPipeline,
                    model_name="3.0.0",
                    tokenizer=tokenizer,
                )
            question = eval_item["question"]
            answer = eval_item["answer"]
            model_answer = eval_item["generated_text"]
            text_input = TEMPLATES['extract_answer_0_shot'].format(question=question, ground_truth=answer,
                                                                   model_response=model_answer)
            text_input = self.extractor.tokenizer.apply_chat_template([{"role": "user", "content": text_input}], tokenize=False,
                                                       add_generation_prompt=True)
            outputs = self.extractor(text_input, num_return_sequences=1, max_new_tokens=50, num_beams=5)
            model_answer = outputs[0]["generated_text"]



        else:
            model_answer = eval_item["generated_text"]

            if self.chain_of_confidence:
                ori_answer = model_answer
                pattern = r"final answer:? is? (.+)"
                pattern = r"Final answer: (.+)"
                pattern = re.compile(pattern, re.IGNORECASE)
                match = pattern.search(model_answer)
                if match:
                    # Extracted part after "Final answer:"
                    model_answer = match.group(1)
                    # print("Extracted final answer:", model_answer, ori_answer)

        if self.dataset_name == "evaldoc":
            model_answer = "correct" if get_answer_evaldoc(model_answer) else "wrong"
        elif self.dataset_name == "misleadqa_fc":
            model_answer = get_answer_misleadqa_fc(model_answer)
        # if self.multiple_choice:
        #     model_answer = extract_answer(eval_item)
        return model_answer

    def get_self_consistency(self, eval_item, norm_method=""):
        question, generated_text = eval_item["question"], eval_item["generated_text"]
        model_answer = self.get_model_answer(eval_item, norm_method, False)
        consistency_scores = []
        for other_answer in eval_item["other_answers"]:
            eval_item_ = deepcopy(eval_item)
            eval_item_["generated_text"] = other_answer
            other_answer_ = self.get_model_answer(eval_item_, norm_method)
            consistency_scores.append(f1_score_token_level(model_answer, other_answer_))
        return np.mean(consistency_scores)


    def process_predictions(self, norm_method=""):

        self.predictions_df["model_answer"] = self.predictions_df.apply(self.get_model_answer,
                                                                        args=[norm_method, True], axis=1)
        self.predictions_df["model_other_answers"] = self.predictions_df.apply(
            lambda x: [self.get_model_answer({"generated_text": x}, norm_method) for x in x["other_answers"]], axis=1)

        return self.predictions_df

    def calculate_ece_score(self, confidence_scores, correctness_scores, n_bins=10):
        # calculate ece score
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_boundaries[-1] += 1e-4
        ece_score = 0.0
        total_sample = len(confidence_scores)
        total_props = 0
        for bin_idx in range(n_bins):
            bin_mask = (confidence_scores >= bin_boundaries[bin_idx]) & (confidence_scores < bin_boundaries[bin_idx + 1])
            bin_confidence_scores = confidence_scores[bin_mask]
            bin_correctness_scores = correctness_scores[bin_mask]
            if len(bin_confidence_scores) == 0:
                continue
            bin_prop = len(bin_confidence_scores) / total_sample
            total_props += bin_prop
            bin_err = np.abs(np.mean(bin_correctness_scores) - np.mean(bin_confidence_scores))
            ece_score += bin_err * bin_prop
        return ece_score


    def evaluate(self, norm_method="", sc_norm_method=""):
        self.multiple_choice = True if "multiple_choice" in self.prediction_file else False
        # if "triviaqa" in self.prediction_file or self.prediction_file == "":
        self.predictions_df = self.process_predictions(norm_method=norm_method)
        if "triviaqa" in self.prediction_file or self.prediction_file == "":
            evaluate_func = evaluate_triviaqa_df if not self.multiple_choice else evaluate_multiple_choice
        elif "evaldoc" in self.prediction_file:
            evaluate_func = evaluate_evaldoc
        elif "misleadqa_fc" in self.prediction_file:
            evaluate_func = evaluate_misleadqa_fc
        else:
            raise NotImplementedError()
        total_scores, scores = evaluate_func(self.predictions_df)
        self.predictions_df["total_scores"] = [total_scores] * len(self.predictions_df)
        self.predictions_df["scores"] = scores

        confidence_methods = ["self_consistency"]
        self.predictions_df["self_consistency"] = self.predictions_df.apply(self.get_self_consistency,
                                                                            args=[sc_norm_method], axis=1)
        if self.recalibration_model:
            self.predictions_df["self_consistency"] = self.recalibration_model.transform(self.predictions_df["self_consistency"])
        if "seq_log_prob_average" in self.predictions_df.columns:
            self.predictions_df["seq_prob"] = self.predictions_df.apply(lambda x: np.exp(x["seq_log_prob"]), axis=1)
            self.predictions_df["seq_prob_avg"] = self.predictions_df.apply(lambda x: np.exp(x["seq_log_prob_average"]), axis=1)
            self.predictions_df["seq_prob_filtered"] = self.predictions_df.apply(lambda x: np.exp(x["seq_log_prob_filtered"]), axis=1)
            confidence_methods = ["self_consistency", "seq_prob", "seq_prob_avg", "seq_prob_filtered"]

        # self.predictions_df["correctness_score"] = self.predictions_df.apply(lambda x: x["scores"]["recall"], axis=1)
        if "triviaqa" in self.prediction_file:
            self.predictions_df["correctness_score"] = self.predictions_df.apply(lambda x: x["scores"]["em_relax"],
                                                                                 axis=1)
            self.predictions_df["expected_correctness"] = self.predictions_df.apply(
                lambda x: evaluate_triviaqa_row(x), axis=1)
        elif "evaldoc" in self.prediction_file:
            self.predictions_df["correctness_score"] = self.predictions_df["scores"]
        elif "misleadqa_fc" in self.prediction_file:
            self.predictions_df["correctness_score"] = self.predictions_df["scores"]
            self.predictions_df["expected_correctness"] = self.predictions_df.apply(
                lambda x: evaluate_misleadqa_fc_row(x), axis=1
            )
        else:
            raise NotImplementedError()
        self.predictions_df["confidence_score"] = self.predictions_df["self_consistency"]

        # df = self.predictions_df[["question", "answer", "generated_text", "model_answer", "correctness_score", "confidence_score", "seq_prob", "seq_prob_avg", "seq_prob_filtered"]]
        ece_score = self.calculate_ece_score(self.predictions_df["confidence_score"].values, self.predictions_df["correctness_score"].values)
        self.predictions_df["ece_score"] = [ece_score] * len(self.predictions_df)

        for confidence_method in confidence_methods:
            ece_score = self.calculate_ece_score(self.predictions_df[confidence_method].values, self.predictions_df["correctness_score"].values)
            print(f"ece_score for {confidence_method}: {ece_score}")
        return total_scores



def main(prediction_file, recalibration_model_path=None):
    evaluator = Evaluator(prediction_file, recalibration_model_path)

    scores = evaluator.evaluate()
    print(scores)

    df = evaluator.predictions_df
    save_jsonl(df, prediction_file + ".score")



if __name__ == "__main__":
    fire.Fire(main)

