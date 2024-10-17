import pandas as pd
import transformers
import torch
import fire
import os
import json
import numpy as np

from pipeline import (MyPipeline, pipeline_init, RestrictTokensLogitsProcessor, CoTSituatedFaithfulQAPipeline)
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from omegaconf import OmegaConf
from tqdm import tqdm
from prompter import Prompter
from utils import read_jsonl, save_jsonl, OPENAI_MODELS, multi_process_map
from datasets import Dataset
from eval import Evaluator
import pandas as pd



def main(
        config_path="configures/v0.0.0.yml"
):
    args = OmegaConf.load(config_path)

    print(args)
    model = args.model
    model_name = args.model.split("/")[-1] if args.get("model_name", None) is None else args.model_name

    if args.get("model_path", None) is not None:
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model) if model not in OPENAI_MODELS else None


    eos_token_id = tokenizer.eos_token_id if model not in OPENAI_MODELS else 0
    confidence_method = args.get("confidence_method", "")
    confidence_to_pipeline = {
        "": MyPipeline,
        "cot_situated": CoTSituatedFaithfulQAPipeline,
    }

    num_return_sequences = args.get("num_return_sequences", 1)

    pipeline = pipeline_init(
        task="text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        pipeline_class=confidence_to_pipeline[confidence_method],
        model_name=model_name,
        tokenizer=tokenizer,
    )

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load data
    eval_data = read_jsonl(args.eval_file)

    # sample part of the data
    if args.sample_size > 0:
        sample_start = args.get("sample_start", 0)
        if "is_doc_correct" not in eval_data.columns:
            eval_data = eval_data[sample_start: sample_start + args.sample_size]
            print(f"sample from {sample_start} to {sample_start + args.sample_size}")
        # if the data include both correct and incorrect doc, we sample the same portion of correct and incorrect doc
        else:
            half_sample_size = args.sample_size // 2
            eval_data_correct_doc = eval_data[eval_data["is_doc_correct"] == 1].reset_index(drop=True)[sample_start: sample_start + half_sample_size]
            eval_data_incorrect_doc = eval_data[eval_data["is_doc_correct"] == 0].reset_index(drop=True)[sample_start: sample_start + half_sample_size]
            assert all(eval_data_correct_doc["question"] == eval_data_incorrect_doc["question"])
            if "__index_level_0__" in eval_data_correct_doc.columns:
                eval_data_correct_doc.drop(columns=[ "__index_level_0__"], inplace=True)
            if "__index_level_0__" in eval_data_incorrect_doc.columns:
                eval_data_incorrect_doc.drop(columns=[ "__index_level_0__"], inplace=True)
            eval_data = pd.concat([eval_data_correct_doc, eval_data_incorrect_doc])
            print(len(eval_data))
            print(f"sample from {sample_start} to {sample_start + half_sample_size} with correct doc and incorrect doc")

    else:
        args.sample_size = len(eval_data)

    prompter_ = Prompter(
        n_shot=args.n_shot,
        n_doc=args.n_doc,
        n_doc_in_demo=args.get("n_doc_in_demo", 0),
        no_doc_in_demo=args.get("no_doc_in_demo", True),
        dataset_name=args.dataset_name,
        model_name=args.model,
        demo_prompt_idx=args.get("demo_prompt_idx", None),
    )


    max_new_tokens = args.get("max_new_tokens", 512)
    prompter_task_type = args.get("task_type", "main")
    logits_processor = None
    if args.get("num_options", 0) > 0:
        allowed_tokens = [chr(ord('A') + i) for i in range(args.num_options)]
        allowed_tokens_preprocessor = RestrictTokensLogitsProcessor(tokenizer, allowed_tokens=allowed_tokens)
        logits_processor = [allowed_tokens_preprocessor]
        max_new_tokens = 1
        prompter_task_type = "multiple_choice"


    temperature = args.get("temperature", 0.6)

    def get_model_answer(eval_item):
        # turn df to dict if needed
        if isinstance(eval_item, pd.Series):
            eval_item = eval_item.to_dict()
        text_input = prompter_.generate_text_input(task_type=prompter_task_type,
                                                   eval_item=eval_item,
                                                   faithful_type=args.get("faithful_type", None),)

        eval_item['text_input'] = text_input

        # if idx == 0:
        #     print(text_input)


        if args.get("answer_file", None) is None:
            sequences = pipeline(
                text_input,
                do_sample=True if temperature > 0 else False,
                top_k=10,
                num_return_sequences=num_return_sequences,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
                # max_length=2048,
                max_new_tokens=max_new_tokens,
                random_state=args.seed,
                temperature=temperature,
                logits_processor=logits_processor,
                prompter=prompter_,
                eval_item=eval_item,
            )

            eval_item.update(sequences[0])
            other_answers = [item["generated_text"] for item in sequences[1:]]
        else:
            other_answers = None

        eval_item["other_answers"] = other_answers
        return eval_item


    # run the model for inference, for gpt model, we can use multi-process to speed up
    if args.get("multi_process", False):
        eval_data = multi_process_map(eval_data, get_model_answer, num_proc=64)
    else:
        eval_data_output = []
        for idx, eval_item in tqdm(eval_data.iterrows()):
            eval_data_output.append(get_model_answer(eval_item))
        eval_data = pd.DataFrame(eval_data_output)




    save_dir = os.path.join(args.save_dir, f"results/{args.exp_name}/{args.dataset_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"create {save_dir}")
    save_path = os.path.join(save_dir, f"{model_name}_predictions")


    sample_suffix = f"_{args.sample_start}:{args.sample_start + args.sample_size}" \
        if args.get("sample_start", None) is not None else f"_{args.sample_size}"
    n_shot_suffix = f"_{args.n_shot}-shot"
    temperature_suffix = f"_t:{temperature}" if temperature != 0.6 else ""
    save_path = save_path + n_shot_suffix + temperature_suffix + sample_suffix + ".jsonl"
    if args.get("save_path", None) is not None:
        save_path = args.save_path

    save_jsonl(eval_data, save_path)

    # evaluate the model if needed
    if args.get("do_eval", True):
        prediction_file = save_path
        evaluator = Evaluator(prediction_file)

        scores = evaluator.evaluate()
        print(scores)

        df = evaluator.predictions_df
        save_jsonl(df, prediction_file + ".score")

if __name__ == '__main__':
    fire.Fire(main)