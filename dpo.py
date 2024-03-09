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
from trl import SFTTrainer, DPOTrainer


from accelerate import Accelerator
from datasets import Dataset
from prompter import Prompter



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

    # LoRA Config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # Model to fine-tune
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    model.config.use_cache = False

    # Reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )

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
        optim="paged_adamw_32bit",
        warmup_steps=conf.get("warmup_steps", 0),
        bf16=True,
        gradient_accumulation_steps=conf.get("gradient_accumulation_steps", 1),

    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
    )

    # Fine-tune model with DPO
    dpo_trainer.train()

    save_dir = os.path.join(save_dir, "checkpoint-final")
    print(f"Saving last checkpoint of the model to {save_dir}")
    dpo_trainer.model.save_pretrained(save_dir)

    wandb.finish()




if __name__ == '__main__':
    fire.Fire(main)
