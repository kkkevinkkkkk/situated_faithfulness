from transformers import pipeline, TextGenerationPipeline, AutoModelForTokenClassification, AutoTokenizer
import torch.nn.functional as F
from transformers.pipelines.text_generation import ReturnType
import torch
import transformers
import openai
import os
import re
from retry import retry
from openai.error import Timeout, APIError, ServiceUnavailableError, InvalidRequestError, APIConnectionError
from omegaconf import DictConfig
from nltk.corpus import stopwords
from nltk import sent_tokenize
from string import punctuation
import numpy as np
from transformers import LogitsProcessor


from collections import defaultdict

from prompter import Prompter
from utils import CURRENT_DATE, OPENAI_MODELS

from omegaconf import OmegaConf

# OPENAI_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o", "ft:gpt-4o-2024-08-06:duke-university:faithful-93:A3S28crx"]
openai.api_key = os.getenv("OPENAI_API_KEY")


def pipeline_init(**kwargs):
    if kwargs["model"] in OPENAI_MODELS:
        if kwargs["model"] == "gpt-4":
            kwargs["model"] = "gpt-4-0125-preview"
            kwargs["model_name"] = "gpt-4-0125-preview"
            print("use gpt-4-turbo")
        return kwargs["pipeline_class"](**kwargs)
    else:
        return transformers.pipeline(**kwargs)

def Record(generated_text=None,
           seq_log_prob=0,
           full_text="",
           **kwargs):
    d = {
        "generated_text": generated_text,
        "seq_log_prob": seq_log_prob,
        "full_text": full_text
    }
    d.update(kwargs)
    return d


class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_tokens):
        self.allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in allowed_tokens]

    def __call__(self, input_ids, scores):
        # Set logits for all tokens not in the allowed list to a large negative value
        disallowed_token_mask = torch.ones_like(scores).bool()
        disallowed_token_mask[:, self.allowed_token_ids] = False
        scores[disallowed_token_mask] = -float('Inf')
        return scores

class MyPipeline(TextGenerationPipeline):
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs["model_name"] if "model_name" in kwargs else kwargs["model"]
        self.openai = True if self.model_name in OPENAI_MODELS else False

        def is_trained_model(s):
            pattern = r'^\d+\.\d+\.\d+$'
            return bool(re.match(pattern, s))

        if "vicuna" in self.model_name:
            self.model_type = "vicuna"
        elif "chat" in self.model_name or is_trained_model(self.model_name):
            self.model_type = "chat"
        else:
            self.model_type = "openai"

        if not self.openai:
            super().__init__(*args, **kwargs)
            self._forward_params = {}

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                    "generation_config" in generate_kwargs
                    and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                    "generation_config" in generate_kwargs
                    and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length
        generate_kwargs["output_scores"] = True
        generate_kwargs["return_dict_in_generate"] = True
        # generation_config = GenerationConfig(**generate_kwargs)
        # BS x SL
        # generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        scores = generated_sequence["scores"]
        generated_sequence = generated_sequence["sequences"]
        seq_tokens = generated_sequence[:, -len(scores):]
        scores_tokens = [(F.log_softmax(scores[t], dim=-1), seq_tokens[:, t]) for t in range(len(scores))]
        sampled_scores = [
            scores_t.gather(-1, token_t.unsqueeze(-1))
            for scores_t, token_t in scores_tokens
        ]
        sampled_scores = torch.cat(sampled_scores, dim=1)

        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        return {"generated_sequence": generated_sequence, "input_ids": input_ids,
                "prompt_text": prompt_text, "scores": sampled_scores}

    @staticmethod
    def post_process_cal_logprob(tokens_probs):
        filtered_cnt = 0
        filtered_logprob_sum = 0
        logprob_sum = 0
        cnt = 0
        for (token, prob) in tokens_probs:
            if token not in stopwords.words("english") and token != "<unk>" and token not in punctuation:
                filtered_logprob_sum += np.log(prob)
                filtered_cnt += 1
            if token != "<unk>":
                logprob_sum += np.log(prob)
                cnt += 1
        filtered_logprob = filtered_logprob_sum / filtered_cnt if filtered_cnt > 0 else 0
        logprob = logprob_sum / cnt if cnt > 0 else 0
        return {"seq_log_prob_average": logprob, "seq_log_prob_filtered": filtered_logprob}

    def postprocess(self, model_outputs, return_type=ReturnType.NEW_TEXT, clean_up_tokenization_spaces=True):
        generated_sequence = model_outputs["generated_sequence"][0]
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]
        scores = model_outputs["scores"]
        tokens_probs_list = [[(self.tokenizer.decode(token), torch.exp(score).item())
                              for score, token in zip(scores[i], generated_sequence[i][-len(scores[i]):])] for i in
                             range(len(scores))]

        scores_copy = scores.clone()
        scores_copy[scores_copy == -float('inf')] = 0
        seq_log_probs = torch.sum(scores_copy, dim=-1)

        generated_sequence = generated_sequence.numpy().tolist()
        records = []
        for i, sequence in enumerate(generated_sequence):
            if return_type == ReturnType.TENSORS:
                record = {"generated_token_ids": sequence}
            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # Decode text
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )

                all_text = text[prompt_length:]
                full_text = prompt_text + all_text
                if return_type == ReturnType.FULL_TEXT:
                    all_text = prompt_text + all_text

                # record = {"generated_text": all_text, "tokens_probs": tokens_probs_list[i],
                #           "seq_log_prob": seq_log_probs[i].item(), "full_text": full_text}
                record = {"generated_text": all_text, "seq_log_prob": seq_log_probs[i].item(), "full_text": full_text,
                          "output": all_text}
                log_probs = self.post_process_cal_logprob(tokens_probs_list[i])
                record.update(log_probs)
                record = Record(**record)
            records.append(record)

        return records

    @retry((Timeout, APIError, ServiceUnavailableError, InvalidRequestError, APIConnectionError), tries=50, delay=1, backoff=6, max_delay=300)
    def get_openai_completion(self, prompt, temperature=1.0, model_name=None):
        messages = [{"role": "user", "content": prompt}]
        model_name = model_name if model_name is not None else self.model_name

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature, )
        output = response.choices[0].message["content"]

        record = Record(**{"generated_text": output,
                           "seq_log_prob": 0, "full_text": prompt + "\n" + output,
                           "output": output})
        records = [record]
        return records

    @staticmethod
    def get_openai_completion_static(prompt, temperature=1.0, model_name=None):
        messages = [{"role": "user", "content": prompt}]

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature, )
        output = response.choices[0].message["content"]

        record = Record(**{"generated_text": output,
                           "seq_log_prob": 0, "full_text": prompt + "\n" + output,
                           "output": output})
        records = [record]
        return records

    @staticmethod
    def extract_score(text, pattern=r"(\d+)/100"):
        match = re.search(pattern, text)
        if not match:
            print("Warning!!!", text)
        score = int(match.group(1)) if match else 0
        return score

    def call_once(self, inputs, *args, temperature=0.6,
                 num_workers=None, batch_size=None, random_state=1, **kwargs):

        if self.openai:
            return self.get_openai_completion(inputs, temperature=temperature)
        else:
            kwargs["temperature"] = temperature
            torch.cuda.manual_seed(random_state)
            records = super().__call__(inputs, **kwargs)
            return records

    def _remove_irrelevant_kwargs(self, kwargs):
        irrelevant_generation_kwargs = ["prompter", "eval_item"]
        for key in irrelevant_generation_kwargs:
            kwargs.pop(key, None)
        return kwargs
    def __call__(self, inputs, *args, temperature=0.6,
                 num_workers=None, batch_size=None, num_return_sequences=1, random_state=1, **kwargs):
        sequences = []
        for i in range(num_return_sequences):
            kwargs = self._remove_irrelevant_kwargs(kwargs)
            outputs = self.call_once(inputs, *args, temperature=temperature,
                                     num_workers=num_workers, batch_size=batch_size,
                                     num_return_sequences=1, random_state=random_state + i, **kwargs)
            sequences.append(outputs[0])
        return sequences

    def extract_confidence_score(self, answer, question, dataset_name="asqa", temperature=0, **kwargs):
        output = {}
        return output

    @staticmethod
    def extract_confidence_distribution(scores, category_num=6):
        confidence_distribution = [0] * category_num
        score_interval = 100 / (category_num - 1)
        for score in scores:
            if np.isnan(score):
                score = 0
            confidence_distribution[int(round(score / score_interval))] += 1 / len(scores)
        confidence_distribution = [round(confidence_distribution[i], 3) for i in range(category_num)]
        return confidence_distribution



class EvaluationPipeline(MyPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def evaluate_answer(self, answer, question, gold_answer, dataset_name="asqa",
                        use_examples=True, five_pnt=False, temperature=0,
                        **kwargs):
        inputs = Prompter(model_name=self.model_name, dataset_name=dataset_name).generate_text_input(
            question=question, answer=answer, reference_answer=gold_answer, task_type="eval", **kwargs)

        outputs = self.__call__(inputs, temperature=temperature, **kwargs)
        comment = outputs[0]["generated_text"]
        if five_pnt:
            score = self.extract_score(comment, pattern=r"Score: (\d+)/5") * 20
        else:
            score = self.extract_score(comment)


        fact_inputs = ""
        fact_comment = ""
        fact_score = 0

        output = {
            "inputs_eval": inputs,
            "comment": comment,
            "score": score,
            "fact_inputs_eval": fact_inputs,
            "fact_comment": fact_comment,
            "fact_score": fact_score
        }
        return output


class HybridSituatedFaithfulQAPipeline(MyPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs, *args, eval_item=None, prompter=None, temperature=0.6,
                 num_workers=None, batch_size=None, num_return_sequences=1, random_state=1, **kwargs):
        sequences = []
        for i in range(num_return_sequences):
            prompter.demo_prompt_idx = 1
            prompter.n_doc = 0
            inputs = prompter.generate_text_input(eval_item=eval_item, faithful_type=None)
            outputs = self.call_once(inputs, *args, temperature=temperature,
                                     num_workers=num_workers, batch_size=batch_size,
                                     num_return_sequences=1, random_state=random_state + i, **kwargs)
            model_answer = outputs[0]["generated_text"]
            prompter.n_doc = 1
            inputs = prompter.generate_text_input(eval_item=eval_item, faithful_type="complete")
            outputs = self.call_once(inputs, *args, temperature=temperature,
                                     num_workers=num_workers, batch_size=batch_size,
                                     num_return_sequences=1, random_state=random_state + i, **kwargs)
            evidence_answer = outputs[0]["generated_text"]

            instruction = "You will be given a multiple-choice question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer."
            if prompter.dataset_name == "freshqa":
                instruction = f"You will be given a multiple-choice question and a document. The document may not be trustworthy and the question might be based on false premises. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer as the date of {CURRENT_DATE}."
            TEMPLATE_MULTIPLE_CHOICE = '''{instruction}

Question: {question}

Choices:
{choices}

Document: {document}


Return your answer in the following format:
choice letter) answer1'''

            choices = f"A) {model_answer}\nB) {evidence_answer}\n"
            # choices = f"A) {evidence_answer}\nB) {model_answer}\n"
            input_prompt = TEMPLATE_MULTIPLE_CHOICE.format(
                question=eval_item["question"],
                document=eval_item["docs"][0]["text"],
                choices=choices,
                instruction=instruction
            )
            messages = [{"input": input_prompt}]
            input_prompt = prompter.process_messages(messages)

            outputs = self.call_once(input_prompt, *args, temperature=temperature,
                                           num_workers=num_workers, batch_size=batch_size,
                                           num_return_sequences=1, random_state=random_state + i, **kwargs)
            final_answer = outputs[0]["generated_text"]
            final_answer = re.sub(r"[A-Z]\)", "", final_answer).strip()
            outputs[0]['generated_text'] = final_answer
            outputs[0]['output'] = final_answer
            outputs[0]['internal_answer'] = model_answer
            outputs[0]['evidence_answer'] = evidence_answer
            # outputs[0]['generated_text'] = evidence_answer
            # outputs[0]['output'] = evidence_answer
            sequences.append(outputs[0])

        return sequences

class MC2Pipeline(MyPipeline):
    def __call__(self, inputs, *args, eval_item=None, prompter=None, temperature=0.6,
                 num_workers=None, batch_size=None, num_return_sequences=1, random_state=1, **kwargs):
        sequences = []
        for i in range(num_return_sequences):
            model_answer = eval_item["internal_answer"]
            evidence_answer = eval_item["model_doc_answer"]

            instruction = "You will be given a multiple-choice question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer."
            if prompter.dataset_name == "freshqa":
                instruction = f"You will be given a multiple-choice question and a document. The document may not be trustworthy and the question might be based on false premises. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer as the date of {CURRENT_DATE}."
            TEMPLATE_MULTIPLE_CHOICE = '''{instruction}

    Question: {question}

    Choices:
    {choices}

    Document: {document}


    Return your answer in the following format:
    choice letter) answer1'''

            choices = f"A) {model_answer}\nB) {evidence_answer}\n"
            # choices = f"A) {evidence_answer}\nB) {model_answer}\n"
            input_prompt = TEMPLATE_MULTIPLE_CHOICE.format(
                question=eval_item["question"],
                document=eval_item["docs"][0]["text"],
                choices=choices,
                instruction=instruction
            )
            messages = [{"input": input_prompt}]
            input_prompt = prompter.process_messages(messages)

            outputs = self.call_once(input_prompt, *args, temperature=temperature,
                                     num_workers=num_workers, batch_size=batch_size,
                                     num_return_sequences=1, random_state=random_state + i, **kwargs)
            final_answer = outputs[0]["generated_text"]
            final_answer = re.sub(r"[A-Z]\)", "", final_answer).strip()
            outputs[0]['generated_text'] = final_answer
            outputs[0]['output'] = final_answer
            outputs[0]['internal_answer'] = model_answer
            outputs[0]['model_doc_answer'] = evidence_answer
            # outputs[0]['generated_text'] = evidence_answer
            # outputs[0]['output'] = evidence_answer
            sequences.append(outputs[0])

        return sequences

class HybridCoTSituatedFaithfulQAPipeline(MyPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs, *args, eval_item=None, prompter=None, temperature=0.6,
                 num_workers=None, batch_size=None, num_return_sequences=1, random_state=1, **kwargs):
        sequences = []
        for i in range(num_return_sequences):
            model_answer = eval_item["internal_answer"]
            evidence_answer = eval_item["model_doc_answer"]


            instruction = "You will be given a multiple-choice question and a document. The document may not be trustworthy. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer."
            if prompter.dataset_name == "freshqa":
                instruction = f"You will be given a multiple-choice question and a document. The document may not be trustworthy and the question might be based on false premises. Use your judgment to assess the reliability of the document. Then, based on both your assessment and your own knowledge, provide the best possible answer as the date of {CURRENT_DATE}."
            TEMPLATE_MULTIPLE_CHOICE = '''{instruction}

Question: {question}

Choices:
{choices}

Document: {document}


Return your answer in the following format:
choice letter) answer1'''

            choices = f"A) {model_answer}\nB) {evidence_answer}\n"
            # choices = f"A) {evidence_answer}\nB) {model_answer}\n"
            input_prompt = TEMPLATE_MULTIPLE_CHOICE.format(
                question=eval_item["question"],
                document=eval_item["docs"][0]["text"],
                choices=choices,
                instruction=instruction
            )
            messages = [{"input": input_prompt}]
            input_prompt = prompter.process_messages(messages)

            outputs = self.call_once(input_prompt, *args, temperature=temperature,
                                           num_workers=num_workers, batch_size=batch_size,
                                           num_return_sequences=1, random_state=random_state + i, **kwargs)
            final_answer = outputs[0]["generated_text"]
            final_answer = re.sub(r"[A-Z]\)", "", final_answer).strip()
            outputs[0]['generated_text'] = final_answer
            outputs[0]['output'] = final_answer
            outputs[0]['internal_answer'] = model_answer
            outputs[0]['model_doc_answer'] = evidence_answer
            # outputs[0]['generated_text'] = evidence_answer
            # outputs[0]['output'] = evidence_answer
            sequences.append(outputs[0])

        return sequences




class LLM:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.openai = False
        if model_name in OPENAI_MODELS:
            self.openai = True
            self.pipe = pipeline_init(
                task="text-generation",
                model=model_name,
                pipeline_class=MyPipeline,
            )
        else:
            pipeline_class = TextGenerationPipeline
            self.pipe = pipeline_init(
                task="text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                pipeline_class=pipeline_class,
            )

        self.tokenizer = self.pipe.tokenizer if not self.openai else None

    def __call__(self, text_input, *args, **kwargs):
        messages = []
        prompt = ""
        if "demos" in kwargs:
            demos = kwargs["demos"]
            for demo in demos:
                messages.extend([
                    {"role": "user", "content": demo["input"]},
                    {"role": "assistant", "content": demo["output"]},
                ])
                if self.openai:
                    prompt += demo["input"] + demo["output"] + "\n"

        if self.openai:
            prompt += text_input
            # print(prompt)
            outputs = self.pipe(prompt)
            return outputs

        messages.extend([
            {"role": "user", "content": text_input},
        ])
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.pipe(prompt,
                            max_new_tokens=256,
                            eos_token_id=terminators,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9, )
        return outputs

GPT4 = LLM(model_name="gpt-4-0125-preview")
GPT3_5 = LLM(model_name="gpt-3.5-turbo")
GPT4o = LLM(model_name="gpt-4o")
GPT4o_mini = LLM(model_name="gpt-4o-mini")