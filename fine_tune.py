from omegaconf import OmegaConf
import fire
import torch
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd

import os
import wandb
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


from accelerate import Accelerator
from datasets import Dataset
from prompter import Prompter

def process_data(data, dataset_name):
    dataset = []
    prompter_ = Prompter(model_name="meta-llama/Llama-2-13b-chat-hf", dataset_name=dataset_name)

    for i, eval_item in data.iterrows():

        prompt = prompter_.generate_text_input(task_type="self_eval", question=eval_item['question'], answer=eval_item['generated_text'])
        prompt += "\n" + eval_item['gpt-4_comment'] + ' </s>'
        dataset.append({"text": prompt})
    dataset_df = pd.DataFrame(dataset)
    dataset = Dataset.from_pandas(dataset_df)
    return dataset

def main(
        config_path="configures/v0.0.0.yml"
):
    conf = OmegaConf.load(config_path)
    wandb.init(
        # set the wandb project where this run will be logged
        project="train_self_eval",
        # track hyperparameters and run metadata
        config={k:v for k, v in conf.items()},
        name=conf.version,
    )
    # set seed
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)


    model_name = conf.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().process_index},
    )
    model.config.use_cache = False
    # More info: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1
    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # train_data = pd.read_json(os.path.join(conf.save_dir, conf.dataset_name,"train_v1.0.json"))
    # val_data = pd.read_json(os.path.join(conf.save_dir, conf.dataset_name,"val_v1.0.json"))

    train_data = pd.read_json(conf['train_data_path'])
    val_data = pd.read_json(conf['val_data_path'])


    if conf.sample_num > 0:
        train_data = train_data[:conf.sample_num]
        val_data = val_data[:conf.sample_num]

    # train_dataset = process_data(train_data, conf.dataset_name)
    # val_dataset = process_data(val_data, conf.dataset_name)

    # shuffle data
    train_data = train_data.sample(frac=1, random_state=conf.seed)
    val_data = val_data.sample(frac=1, random_state=conf.seed)

    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    save_dir = os.path.join(conf.save_dir, conf.version)

    # Define Trainer
    args = TrainingArguments(
        output_dir=os.path.join(save_dir),
        evaluation_strategy="steps",
        eval_steps=conf.eval_steps,
        save_steps=conf.save_steps,
        learning_rate=conf.learning_rate,
        per_device_train_batch_size=conf.per_device_train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=conf.num_train_epochs,
        seed=conf.seed,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=10,
        run_name=conf.version,
        report_to="wandb",
        warmup_steps=conf.get("warmup_steps", 0),
        gradient_accumulation_steps=conf.get("gradient_accumulation_steps", 1),
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=args,
        max_seq_length=conf.get("max_length", 2048),
    )

    trainer.train()
    save_dir = os.path.join(save_dir, "checkpoint-final")
    print(f"Saving last checkpoint of the model to {save_dir}")
    trainer.model.save_pretrained(save_dir)

    wandb.finish()




if __name__ == '__main__':
    fire.Fire(main)
