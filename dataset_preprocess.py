from prompter import Prompter
from pipeline import pipeline_init, MyPipeline
import torch
import pandas as pd
import fire
from utils import read_jsonl, Retriever, NLIModel, save_jsonl, TEMPLATES
from tqdm import tqdm
from copy import deepcopy
from evaluation import f1_score_token_level
import numpy as np

class DatasetPreprocessor:
    def __init__(self, df,
                 evidence_path="/usr/xtmp/yh386/faithfulness/datasets/triviaqa/triviaqa-rc/evidence"):
        tqdm.pandas()
        self.df = df
        self.convert_pipe = None
        self.prompter = None

        self.retriever = None
        self.nli_model = None

        model_name = "gpt-4-0125-preview"
        self.gpt4 = pipeline_init(
            task="text-generation",
            model=model_name,
            pipeline_class=MyPipeline,
        )
        self.evidence_path = evidence_path

    def convert_qa_to_statement_(self, datapoint):
        if self.convert_pipe is None:
            model_name = "meta-llama/Llama-2-13b-chat-hf"
            # model_name = "gpt-3.5-turbo"
            self.convert_pipe = pipeline_init(
                task="text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                pipeline_class=MyPipeline,
                model_name=model_name,
            )
            self.prompter = Prompter(model_name=model_name)

        question = datapoint["Question"]
        answer = datapoint["Answer"]['Value']
        text_input = self.prompter.generate_text_input("qa_to_statement", question=question, answer=answer)
        outputs = self.convert_pipe(text_input)
        return outputs[0]["generated_text"].strip()

    def get_doc_(self, datapoint, method="full"):
        docs = []
        if "EntityPages" in datapoint:
            entity_pages = datapoint["EntityPages"]
            for entity_page in entity_pages:
                if "Filename" in entity_page:
                    filename = entity_page["Filename"]
                    with open(f"{self.evidence_path}/wikipedia/{filename}", 'r') as myfile:
                        data = myfile.read()
                        docs.append(data)
        elif "SearchResults" in datapoint:
            search_results = datapoint["SearchResults"]
            for search_result in search_results:
                if "Filename" in search_result:
                    filename = search_result["Filename"]
                    with open(f"{self.evidence_path}/web/{filename}", 'r') as myfile:
                        data = myfile.read()
                        docs.append(data)
        else:
            print("No evidence found")
        return docs

    def get_support_(self, datapoint, k=10):
        if self.retriever is None:
            self.retriever = Retriever()
        if self.nli_model is None:
            self.nli_model = NLIModel()

        docs = datapoint["raw_docs"]
        qa_statement = datapoint["qa_statement"]
        question = datapoint["Question"]
        support_evidence = []
        qa_id = datapoint["QuestionId"]
        for doc in docs:
            paragraphs = doc.split("\n\n")
            # further split paragraph if it's too long
            long_paragraphs = [p for p in paragraphs if len(p.split()) > 1000]
            short_paragraphs = [p for p in paragraphs if len(p.split()) <= 1000 and len(p.split()) > 20]
            long_paragraphs = [p for para in long_paragraphs for p in para.split("\n")]
            long_paragraphs = [p for p in long_paragraphs if len(p.split()) > 20]
            paragraphs = short_paragraphs + long_paragraphs
            max_length = max([len(p.split()) for p in paragraphs])
            min_length = min([len(p.split()) for p in paragraphs])
            print(f"Question: {question} - Max length: {max_length} - Min length: {min_length} - Paragraphs: {len(paragraphs)}")

            retrieved_paragraphs = self.retriever.retrieve_paragraph(paragraphs, question, k=k, batch_size=32)

            for i, paragraph in enumerate(retrieved_paragraphs):
                # print(f"{qa_id} - {i} - {question} - {len(paragraph.split())}")
                is_evidence = self.nli_model.run(premise=paragraph, hypothesis=qa_statement)
                if is_evidence:
                    support_evidence.append(paragraph)
        find_support = True
        if len(support_evidence) == 0:
            print(f"Warning! Question: {question} has no supporting evidence")
            find_support = False
        support_doc = "\n\n".join(support_evidence[:3]) if len(support_evidence) > 0 else doc
        return support_doc, find_support
    def trim_support(self, datapoint, support_max_length=512):
        support = datapoint["support"]
        find_support = datapoint["find_support"]
        if find_support == False:
            return support
        else:
            if len(support.split()) < support_max_length:
                return support
            evidences = support.split("\n\n")
            trimmed_support = "\n\n".join(evidences[:3])
            current_len = len(trimmed_support.split())
            if current_len > support_max_length:
                trimmed_support = evidences[0]
            # print(f"{len(trimmed_support.split())}")
            return trimmed_support
    def get_gpt4_support_(self, datapoint):
        question = datapoint["Question"]
        answer = datapoint["Answer"]
        find_support = datapoint["find_support"]
        if not find_support:
            text_input = TEMPLATES["generate_support"].format(question=question, answer=answer)
            outputs = self.gpt4(text_input, temperature=1.0)
            support = outputs[0]['output']
        else:
            support = datapoint["support"]
        return support

    def match_format(self):
        self.df["question"] = self.df["Question"]
        self.df["answer"] = self.df.apply(lambda x: x["Answer"]["Value"], axis=1)
        self.df["docs"] = self.df.apply(lambda x: [{"text": x["support"]}], axis=1)
        if "raw_docs" in self.df.columns:
            self.df = self.df.drop(columns=["raw_docs"])
        self.df["document_answer"] = self.df["answer"]

        return self.df

    def generate_multiple_choice_(self, datapoint):
        if self.gpt4 is None:
            self.gpt4 = pipeline_init(
                model="gpt-4",
                torch_dtype=torch.float16,
                device_map="auto",
                pipeline_class=MyPipeline,
                output_scores=True
            )
        question = datapoint["question"]
        answer = datapoint["answer"]
        text_input = TEMPLATES["generate_multiple_choice"].format(question=question, answer=answer)
        outputs = self.gpt4(text_input, temperature=1.0)
        answers = [a.strip() for a in outputs[0]['output'].split(",")]
        return answers
    def generate_deceptive_document_(self, datapoint):
        question = datapoint["question"]
        answer = datapoint["answer"]
        document = datapoint["docs"][0]['text']
        deceptive_answer = datapoint["deceptive_answer"]
        text_input = TEMPLATES["synthesize_deceptive_document"].format(question=question, answer=answer, deceptive_answer=deceptive_answer, document=document)
        outputs = self.gpt4(text_input, temperature=1.0)
        return outputs[0]['output']

    def convert_multiple_choice(self, datapoint, num_options=5):
        candidates = deepcopy(datapoint["multiple_choice"])
        candidates = candidates[:num_options]
        answer = datapoint["answer"]
        # shuffle candidates and convert to multiple choice format, each answer relate to a letter
        np.random.shuffle(candidates)

        candidates_to_letters = {chr(65 + i): candidate for i, candidate in enumerate(candidates)}
        answer_letters = [k for k, v in candidates_to_letters.items() if v == answer]
        if len(answer_letters) == 0:
            # token level match
            f1_scores = [f1_score_token_level(answer, candidate) for candidate in candidates]
            if max(f1_scores) >= 0.5:
                print("Warning! Answer not found in candidates, using token level match to find the answer.")
                answer_letters = [k for k, v in candidates_to_letters.items() if v == candidates[np.argmax(f1_scores)]]
            else:
                raise ValueError(f"Answer: {answer} not found in candidates: {candidates}")

        answer_letter = answer_letters[0]
        muliple_choice_str = "\n".join([f"{k}) {v}" for k, v in candidates_to_letters.items()])
        question = datapoint["Question"] + "\n" + muliple_choice_str
        return question, answer_letter, candidates_to_letters

    def extract_short_docs(self, datapoint):
        question = datapoint["question"]
        reliable_document = datapoint["support"]
        deceptive_document = datapoint["deceptive_document"]
        text_input = TEMPLATES["extract_short_doc"].format(question=question, document=reliable_document)
        reliable_short_doc = self.gpt4(text_input)[0]['generated_text']
        text_input = TEMPLATES["extract_short_doc"].format(question=question, document=deceptive_document)
        deceptive_short_doc = self.gpt4(text_input)[0]['generated_text']
        return reliable_short_doc, deceptive_short_doc



    def preprocess(self,
                   convert_qa_statement=False,
                   retrieve_docs=False,
                   retrieve_support=False,
                   gpt4_support=False,
                   generate_multiple_choice=False,
                   convert_multiple_choice=False,
                   syntheize_deceptive_document=False,
                   extract_short_docs=False,
                   match_format=False):
        if convert_qa_statement:
            self.df["qa_statement"] = self.df.apply(self.convert_qa_to_statement_, axis=1)
            # remove convert_pipe to save memory and remove cache
            self.convert_pipe.model.to("cpu")
            torch.cuda.empty_cache()
            del self.convert_pipe
            self.convert_pipe = None

        if retrieve_docs:
            self.df["raw_docs"] = self.df.apply(self.get_doc_, axis=1)

        if retrieve_support:
            self.df[["support", "find_support"]] = (
                self.df.progress_apply(lambda x: pd.Series(self.get_support_(x, k=50)), axis=1))
            # show how many find_support
            print(f"find_support: {self.df['find_support'].sum()} / {len(self.df)}")
            self.df["support"] = self.df.apply(self.trim_support, axis=1)
            # remove docs column from df
            # self.df = self.df.drop(columns=["raw_docs"])

        if gpt4_support:
            self.df["support"] = self.df.apply(self.get_gpt4_support_, axis=1)
            # remove docs column if exists
            # self.df = self.df.drop(columns=["raw_docs"])

        if generate_multiple_choice:
            self.df["multiple_choice"] = self.df.apply(self.generate_multiple_choice_, axis=1)

        if convert_multiple_choice:
            np.random.seed(1)
            self.df[["question", "answer_letter", "candidates_to_letters"]] = self.df.apply(
                lambda x: pd.Series(self.convert_multiple_choice(x)), axis=1)

        if match_format:
            self.df = self.match_format()

        if syntheize_deceptive_document:
            self.df["deceptive_answer"] = self.df["multiple_choice"].apply(lambda x: x[1])
            self.df["deceptive_document"] = self.df.progress_apply(self.generate_deceptive_document_, axis=1)
            self.df["docs"] = self.df.apply(lambda x: [{"text": x["deceptive_document"]}], axis=1)
            self.df["document_answer"] = self.df["deceptive_answer"]

        if extract_short_docs:
            self.df[["reliable_short_doc", "deceptive_short_doc"]] = self.df.progress_apply(lambda x: pd.Series(self.extract_short_docs(x)), axis=1)



        return self.df



def main(dataset_path, save_path, sample_start=0, sample_size=100,
         convert_qa_statement=False,
         retrieve_docs=False,
         retrieve_support=False,
         gpt4_support=False,
         generate_multiple_choice=False,
         syntheize_deceptive_document=False,
         extract_short_docs=False,
         match_format=False):

    # convert_qa_statement = True
    # retrieve_docs = True
    # retrieve_support = True
    # gpt4_support = True
    # match_format = True
    # generate_multiple_choice = True
    # syntheize_deceptive_document = True
    extract_short_docs = True


    df = read_jsonl(dataset_path)
    # df = df[sample_start:sample_start+sample_size]
    if "id" not in df.columns:
        df["id"] = range(len(df))
        print("setting the id as default")
    df.set_index("id", inplace=True, drop=False)
    df = df.loc[sample_start: sample_start+sample_size-1]
    sample_suffix = f"{sample_start}:{sample_start+sample_size}"

    preprocessor = DatasetPreprocessor(df)
    df = preprocessor.preprocess(convert_qa_statement=convert_qa_statement,
                                 retrieve_docs=retrieve_docs,
                                 retrieve_support=retrieve_support,
                                 gpt4_support=gpt4_support,
                                 generate_multiple_choice=generate_multiple_choice,
                                 syntheize_deceptive_document=syntheize_deceptive_document,
                                 match_format=match_format,
                                 extract_short_docs=extract_short_docs)

    save_path = save_path.split(".")
    save_path = save_path[0] + f"_{sample_suffix}." + save_path[1]
    save_jsonl(df, save_path)
    return



if __name__ == "__main__":
    fire.Fire(main)