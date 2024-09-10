from utils import TEMPLATES, DATASET_PROFILES, make_demo, make_head_prompt, make_demo_messages, OPENAI_MODELS
from transformers import AutoTokenizer

class Prompter:
    def __init__(self, model_name="gpt-3.5-turbo",
                 dataset_name=None,
                 n_shot=0,
                 n_doc=0,
                 n_doc_in_demo=0,
                 fewer_doc_in_demo=False,
                 no_doc_in_demo=True,
                 use_shorter="text",
                 oracle_doc=False,
                 demo_prompt_idx=None,
                 ):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.n_shot = n_shot
        self.n_doc = n_doc
        self.n_doc_in_demo = n_doc_in_demo
        self.fewer_doc_in_demo = fewer_doc_in_demo
        self.no_doc_in_demo = no_doc_in_demo
        self.use_shorter = use_shorter
        self.oracle_doc = oracle_doc
        self.demo_prompt_idx = demo_prompt_idx
        if model_name.startswith("gpt") or model_name in ["meta-llama/Meta-Llama-3-8B"] or model_name in OPENAI_MODELS:
        # if model_name.startswith("gpt"):
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.head_prompt = None

    def generate_main_task_input(self, eval_item=None,
                                 faithful_type=None, **kwargs):
        prompt_data = DATASET_PROFILES[self.dataset_name]
        instruction_key = "instruction" if faithful_type is None else f"instruction_{faithful_type}_faithful"
        demos_key = f"{faithful_type}_faithful_demos" if faithful_type == "cot_situated" else "demos"
        demo_prompt_key = f"demo_prompt_{self.demo_prompt_idx}" if self.demo_prompt_idx else "demo_prompt"

        messages = make_demo_messages(prompt_data,
                                      n_shot=self.n_shot,
                                      n_doc=self.n_doc,
                                      n_doc_in_demo=self.n_doc_in_demo,
                                      fewer_doc_in_demo=self.fewer_doc_in_demo,
                                      no_doc_in_demo=self.no_doc_in_demo,
                                      use_shorter=self.use_shorter,
                                      instruction_key=instruction_key,
                                      demos_key=demos_key,
                                      demo_prompt_key=demo_prompt_key)



        test_input = make_demo(eval_item, prompt_data[demo_prompt_key],
                               doc_prompt=prompt_data["doc_prompt"],
                               instruction=prompt_data[instruction_key],
                               n_doc=self.n_doc,
                               test=True)

        messages.extend([{"input": test_input}])

        return messages

    def process_messages(self, messages):
        if self.tokenizer:
            new_messages = []
            for message in messages:
                new_message = []
                if "input" in message:
                    new_message.append({"role": "user", "content": message["input"]})
                if "output" in message:
                    new_message.append({"role": "assistant", "content": message["output"]})
                new_messages.extend(new_message)
            text_input = self.tokenizer.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True)
        else:
            text_input = ""
            for message in messages:
                if "input" in message:
                    text_input += message["input"]
                if "output" in message:
                    text_input += message["output"] + "\n\n"
        return text_input



    def generate_text_input(self, task_type="main", dataset_name=None, **kwargs):
        alignment_methods = ["chain_of_confidence", "post_editing"]
        messages = []
        text_input = ""
        if task_type == "main" or task_type in alignment_methods or task_type == "multiple_choice":
            messages = self.generate_main_task_input(eval_item=kwargs["eval_item"],
                                                     faithful_type=kwargs["faithful_type"])

        elif task_type == "validate_source":
            text_input = self.generate_demo_input(dataset_name=dataset_name, **kwargs)
        elif task_type == "synthesize_document":
            assert "question" in kwargs and "document" in kwargs and "answer" in kwargs and "wrong_answer" in kwargs
            text_input = TEMPLATES["synthesize_document"].format(question=kwargs["question"],
                                                                 document=kwargs["document"],
                                                                 answer=kwargs["answer"],
                                                                 wrong_answer=kwargs["wrong_answer"])
        elif task_type == "qa_to_statement":
            assert "question" in kwargs and "answer" in kwargs
            text_input = TEMPLATES["qa_to_statement"].format(question=kwargs["question"],
                                                             answer=kwargs["answer"])
        elif task_type == "self_eval":
            eval_item = kwargs["eval_item"]

            text_input = TEMPLATES["self_eval"].format(question=eval_item["question"],
                                                       model_answer=eval_item["model_answer"],
                                                       )

        else:
            raise NotImplementedError
        if len(messages) == 0:
            messages = [{"input": text_input}]


        text_input = self.process_messages(messages)


        return text_input