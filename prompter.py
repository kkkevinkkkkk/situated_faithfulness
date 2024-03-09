from utils import TEMPLATES, DATASET_PROFILES, make_demo, make_head_prompt

class Prompter:
    def __init__(self, model_name="gpt-3.5-turbo",
                 dataset_name=None,
                 n_shot=0,
                 n_doc=0,
                 n_doc_in_demo=0,
                 fewer_doc_in_demo=False,
                 no_doc_in_demo=True,
                 use_shorter="summary",
                 oracle_doc=False,
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

        self.head_prompt = None

    def generate_main_task_input(self, eval_item=None, dataset_name=None, **kwargs):
        prompt_data = DATASET_PROFILES[dataset_name]
        if dataset_name.startswith("triviaqa"):

            post_demo_instruction = prompt_data["post_demo_instruction"]
            if kwargs.get("source_reliability_prompt_idx", None) is not None:
                if kwargs["source_reliability_prompt_idx"] == 0:
                    assert "source_reliability_rate" in kwargs
                    source_reliability_rate = kwargs["source_reliability_rate"]
                    source_reliability_rate = 1.0 if source_reliability_rate is None else source_reliability_rate
                    source_reliability_rate = int(source_reliability_rate * 100)
                    prompt_data["instruction"] += f"The question comes with a document that is {source_reliability_rate}% likely to be correct. "
                    # post_demo_instruction = f"Now let's answer the following question. It comes with a document. And this document is {source_reliability_rate}% correct. \n\n"
                else:
                    pass

            head_prompt = make_head_prompt(prompt_data,
                                           n_shot=self.n_shot,
                                           n_doc=self.n_doc,
                                           n_doc_in_demo=self.n_doc_in_demo,
                                           fewer_doc_in_demo=self.fewer_doc_in_demo,
                                           no_doc_in_demo=self.no_doc_in_demo,
                                           use_shorter=self.use_shorter,
                                           post_demo_instruction=post_demo_instruction)


            text_input = head_prompt + make_demo(eval_item, prompt_data["demo_prompt"],
                                                 doc_prompt=prompt_data["doc_prompt"],
                                                 instruction=None,
                                                 n_doc=self.n_doc,
                                                 test=True)

            if dataset_name.endswith("post_editing"):
                post_test_demo_instruction = 'Step 1: Initial Answer with Confidence Score\nInitial Answer: {model_initial_answer}\nConfidence Score: {model_initial_confidence}%\nStep 2: Document-based Answer\nAnswer According to Document: {document_answer}\nStep 3: Comparative Evaluation and Final Answer\nDocument Confidence: {document_confidence}%\n\nFinal Answer:'

                post_test_demo_instruction =post_test_demo_instruction.format(model_initial_answer=eval_item["model_internal_answer"],
                                                  model_initial_confidence=eval_item["model_internal_confidence"],
                                                  document_answer=eval_item["document_answer"],
                                                  document_confidence=eval_item["docs"][0]["document_confidence"])

                text_input += post_test_demo_instruction

        else:
            raise NotImplementedError

        return text_input
    @staticmethod
    def fit_llama(text_input):
        return TEMPLATES["llama2_chat"].format(task_instruction=text_input)
    def generate_text_input(self, task_type="main", dataset_name=None, **kwargs):
        alignment_methods = ["chain_of_confidence", "post_editing"]
        if task_type == "main" or task_type in alignment_methods:
            dataset_name = dataset_name + f"_{task_type}" if task_type in alignment_methods else dataset_name
            text_input = self.generate_main_task_input(eval_item=kwargs["eval_item"], dataset_name=dataset_name,
                                                       source_reliability_prompt_idx=kwargs["source_reliability_prompt_idx"],
                                                       source_reliability_rate=kwargs["source_reliability_rate"]
                                                       )
        elif task_type == "validate_source":
            text_input = self.generate_demo_input(dataset_name=dataset_name, **kwargs)
        elif task_type == "synthesize_document":
            assert "question" in kwargs and "document" in kwargs and "answer" in kwargs and "wrong_answer" in kwargs
            text_input = TEMPLATES["synthesize_document"].format(question=kwargs["question"],
                                                                 document=kwargs["document"],
                                                                 answer=kwargs["answer"],
                                                                 wrong_answer=kwargs["wrong_answer"])
        elif task_type == "qa_to_statement":
            assert "question" in kwargs  and "answer" in kwargs
            text_input = TEMPLATES["qa_to_statement"].format(question=kwargs["question"],
                                                             answer=kwargs["answer"])
        else:
            raise NotImplementedError

        if "chat" in self.model_name:
            text_input = self.fit_llama(text_input)
            if task_type == "qa_to_statement":
                text_input += "Sure! Statement: "
            # if task_type == "main" and dataset_name == "triviaqa":
            #     text_input += "Sure! The answer to the question is: "

        return text_input