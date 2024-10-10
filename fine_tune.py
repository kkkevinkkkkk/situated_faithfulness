from omegaconf import OmegaConf
import fire
import torch
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from transformers import LlamaForCausalLM
import pandas as pd
import json

import os
import wandb
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import random

from accelerate import Accelerator
from datasets import Dataset
from utils import make_supervised_data_module


def main(
        config_path="configures/v0.0.0.yml"
):
    # get local rank
    device_index = Accelerator().process_index

    conf = OmegaConf.load(config_path)

    use_wandb = True if device_index == 0 else False
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=conf.exp_name,
            # track hyperparameters and run metadata
            config={k:v for k, v in conf.items()},
            name=conf.version,
            group="DDP",
        )
    # set seed
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)


    model_name = conf.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = 'left'

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


    with open(conf['train_data_path'], 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(conf['dev_data_path'], 'r') as f:
        dev_data = [json.loads(line) for line in f]


    if conf.sample_num > 0:
        train_data = train_data[:conf.sample_num]
        dev_data = dev_data[:conf.sample_num]


    # shuffle data
    random.shuffle(train_data)
    random.shuffle(dev_data)

    data_module = make_supervised_data_module(tokenizer=tokenizer, train_data=train_data,
                                              eval_data=dev_data, max_length=conf.get("max_length", 1024))
    save_dir = os.path.join(conf.save_dir, conf.exp_name, conf.version)

    # Define Trainer
    args = TrainingArguments(
        output_dir=os.path.join(save_dir),
        evaluation_strategy="steps",
        eval_steps=conf.eval_steps,
        save_steps=conf.save_steps,
        learning_rate=conf.learning_rate,
        per_device_train_batch_size=conf.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=conf.num_train_epochs,
        seed=conf.seed,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=conf.get("logging_steps", 100),
        run_name=conf.version,
        report_to="wandb" if use_wandb else None,
        warmup_steps=conf.get("warmup_steps", 0),
        gradient_accumulation_steps=conf.get("gradient_accumulation_steps", 1),
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
        peft_config=peft_parameters,
        # dataset_text_field="text",
        tokenizer=tokenizer,
        args=args,
        max_seq_length=conf.get("max_length", 1024),
    )

    trainer.train()
    save_dir = os.path.join(save_dir, "checkpoint-final")
    print(f"Saving last checkpoint of the model to {save_dir}")
    trainer.model.save_pretrained(save_dir)

    wandb.finish()




if __name__ == '__main__':
    fire.Fire(main)
