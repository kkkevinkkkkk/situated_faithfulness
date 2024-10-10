from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class NLIModel:
    def __init__(self, model_name="google/t5_xxl_true_nli_mixture"
, device="cuda"):
        # output hidden states
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.device = device
        self.model.eval()
    def run(self, premise, hypothesis):
        input_text = f"premise: {premise} hypothesis: {hypothesis}"
        # print(input_text)
        # tokenize the input text, truncate if necessary
        # input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True, padding=False,
                                   ).input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=10)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        result = 1 if result.startswith("1") else 0
        return result

