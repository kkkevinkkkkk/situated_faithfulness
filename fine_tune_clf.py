from omegaconf import OmegaConf
import fire
import torch
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer
import pandas as pd
from model import MyLlamaForSequenceClassification
import os
import wandb
from peft import LoraConfig, PeftModel, PeftModelForSequenceClassification
from trl import SFTTrainer
from transformers import DataCollatorWithPadding
import evaluate

from accelerate import Accelerator
from datasets import Dataset
from utils import read_jsonl
from peft import get_peft_model

# PeftModel.save_pretrained()


accuracy = evaluate.load("accuracy")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
def classification_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    acc_class_1 = accuracy.compute(predictions=predictions[labels == 1], references=labels[labels == 1])["accuracy"]
    acc_class_0 = accuracy.compute(predictions=predictions[labels == 0], references=labels[labels == 0])["accuracy"]
    return {"accuracy": acc, "accuracy_class_1": acc_class_1, "accuracy_class_0": acc_class_0}

    # return accuracy.compute(predictions=predictions, references=labels)
def regression_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"mse": ((predictions - labels) ** 2).mean()}
def main(
        config_path="configures/v0.0.0.yml"
):
    conf = OmegaConf.load(config_path)
    wandb.init(
        # set the wandb project where this run will be logged
        project=conf.exp_name,
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

    # process data
    train_data = read_jsonl(conf['train_data_path'])
    dev_data = read_jsonl(conf['dev_data_path'])

    if conf.sample_num > 0:
        train_data = train_data[:conf.sample_num]
        dev_data = dev_data[:conf.sample_num]

    # shuffle data
    train_data = train_data.sample(frac=1, random_state=conf.seed).reset_index(drop=True)
    dev_data = dev_data.sample(frac=1, random_state=conf.seed).reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)

    def preprocess_function(examples):
        # print(tokenizer(examples["text"], truncation=True))
        return tokenizer(examples["text"], truncation=True, max_length=conf.get("max_length", 512))

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    dev_dataset = dev_dataset.map(preprocess_function, batched=True)

    use_lora = conf.get("use_lora", False)

    keep_columns = ["label", "input_ids", "attention_mask"]
    if conf.num_labels == 1:
        # turn float in label into float16
        train_dataset = train_dataset.map(lambda x: {"label": torch.tensor(x["label"], dtype=torch.float16)}, batched=True)
        dev_dataset = dev_dataset.map(lambda x: {"label": torch.tensor(x["label"], dtype=torch.float16)}, batched=True)



    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in keep_columns])
    dev_dataset = dev_dataset.remove_columns([col for col in dev_dataset.column_names if col not in keep_columns])


    if use_lora:
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=False,
        # )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=conf.num_labels,
            torch_dtype=torch.bfloat16,
            # quantization_config=bnb_config,
            device_map={"": Accelerator().process_index},
        )
        peft_parameters = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="SEQUENCE_CLASSIFICATION"
        )

        model = PeftModelForSequenceClassification(model, peft_config=peft_parameters)
        # model = PeftModel(model, peft_parameters)

        # print trainable parameters
        model.print_trainable_parameters()
        # print the name of trainable parameters
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if "score" in name:
        #             param.requires_grad = False


        model.config.use_cache = False

    else:

        model = MyLlamaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=conf.num_labels,
            # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            device_map={"": Accelerator().process_index},
        )
    model.config.pad_token_id = model.config.eos_token_id



    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    save_dir = os.path.join(conf.save_dir, conf.exp_name, conf.version)


    # Define Trainer
    args = TrainingArguments(
        output_dir=os.path.join(save_dir),
        evaluation_strategy="steps",
        eval_steps=conf.eval_steps,
        save_steps=conf.save_steps,
        learning_rate=conf.learning_rate,
        per_device_train_batch_size=conf.per_device_train_batch_size,
        per_device_eval_batch_size=conf.get("per_device_eval_batch_size", 16),
        num_train_epochs=conf.num_train_epochs,
        seed=conf.seed,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=conf.get("logging_steps", 100),
        run_name=conf.version,
        report_to="wandb",
        warmup_steps=conf.get("warmup_steps", 0),
        gradient_accumulation_steps=conf.get("gradient_accumulation_steps", 1),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
        # max_steps=50,
        save_safetensors=False if not use_lora else True,
    )

    compute_metrics = classification_metrics if conf.num_labels > 1 else regression_metrics
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     peft_config=peft_parameters,
    #     train_dataset=train_dataset,
    #     eval_dataset=dev_dataset,
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    #     dataset_text_field="text",
    #     args=args,
    #     compute_metrics=compute_metrics,
    #     # packing=False,
    #     # max_seq_length=conf.get("max_length", 1024),
    # )

    trainer.train()
    save_dir = os.path.join(save_dir, "checkpoint-final")
    print(f"Saving last checkpoint of the model to {save_dir}")
    trainer.model.save_pretrained(save_dir)

    wandb.finish()




if __name__ == '__main__':
    fire.Fire(main)
