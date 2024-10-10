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
from trl import SFTTrainer, DPOTrainer,DPOConfig


from accelerate import Accelerator
from datasets import Dataset
from prompter import Prompter
from utils import read_jsonl
from accelerate import Accelerator


def main(
        config_path="configures/v0.0.0.yml"
):

    device_index = Accelerator().process_index
    device_map = {"": device_index}
    # device_map = "auto"
    conf = OmegaConf.load(config_path)

    use_wandb = True if device_index == 0 else False
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=conf.exp_name,
            # track hyperparameters and run metadata
            config={k: v for k, v in conf.items()},
            name=conf.version,
            group="DDP",
        )

    # set seed
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)


    model_name = conf.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training
    tokenizer.truncation_side = 'left'

    # LoRA Config
    # peft_config = LoraConfig(
    #     lora_alpha=128,
    #     lora_dropout=0.05,
    #     r=256,
    #     bias="none",
    #     # target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    # )
    peft_config = LoraConfig(
        lora_alpha=conf.get("lora_alpha", 16),
        lora_dropout=0.1,
        r=conf.get("r", 8),
        bias="none",
        task_type="CAUSAL_LM"
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


    # Model to fine-tune
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    # Reference model
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     load_in_4bit=True,
    #     device_map=device_map,
    # )

    train_data = read_jsonl(conf['train_data_path'])
    dev_data = read_jsonl(conf.get('dev_data_path', None))


    if conf.sample_num > 0:
        train_data = train_data[:conf.sample_num]
        dev_data = dev_data[:conf.sample_num] if dev_data is not None else None


    # shuffle data
    train_data = train_data.sample(frac=conf.get("sample_frac", 1), random_state=conf.seed)
    dev_data = dev_data.sample(frac=conf.get("sample_frac", 1), random_state=conf.seed) if dev_data is not None else None

    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data) if dev_data is not None else None

    save_dir = os.path.join(conf.save_dir, conf.exp_name, conf.version)

    # Define Trainer
    args = DPOConfig(
        output_dir=os.path.join(save_dir),
        evaluation_strategy="steps",
        eval_steps=conf.eval_steps,
        save_steps=conf.save_steps,
        learning_rate=conf.learning_rate,
        max_grad_norm=conf.max_grad_norm,
        per_device_train_batch_size=conf.per_device_train_batch_size,
        per_device_eval_batch_size=4,
        num_train_epochs=conf.num_train_epochs,
        seed=conf.seed,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=50,
        run_name=conf.version,
        report_to="wandb",
        optim="adamw_torch_fused",
        warmup_steps=conf.get("warmup_steps", 0),
        bf16=True,
        gradient_accumulation_steps=conf.get("gradient_accumulation_steps", 1),
        beta=conf.dpo_beta,
        max_length=conf.max_length,
        max_prompt_length=conf.max_prompt_length,
        rpo_alpha=conf.get("rpo_alpha", None),
    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        # ref_model=ref_model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Fine-tune model with DPO
    dpo_trainer.train()

    save_dir = os.path.join(save_dir, "checkpoint-final")
    print(f"Saving last checkpoint of the model to {save_dir}")
    dpo_trainer.model.save_pretrained(save_dir)

    wandb.finish()




if __name__ == '__main__':
    fire.Fire(main)
