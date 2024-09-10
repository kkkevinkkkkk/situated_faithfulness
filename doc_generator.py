from utils import multi_process_map
from pipeline import GPT4
class DocGenerator:
    def __init__(self, template):
        self.template = template
        self.model = GPT4

    @staticmethod
    def filter_doc(doc):
        doc_splits = doc.split("\n")
        is_filtered = False
        fake_words = ["note", "fabricated", "fake", "false", "incorrect", "wrong", "inaccurate", "unreliable",
                      "misleading", "erroneous", "deceptive", "untruthful", "untrustworthy", "unreliable", "unfounded",
                      "unsubstantiated", "unproven", "hypothetical", "synthesize", "synthetic"
                      "speculate", "speculative", "fictional", "unverified", "unsubstantiated", "unfounded"]
        if any([word in doc_splits[-1].lower() for word in fake_words]):
            doc = "\n".join(doc_splits[:-1])
            is_filtered = True
        return doc, is_filtered

    def generate_deceptive_doc(self, row):
        text_input = self.template.format(question=row["question"], answer=row["doc_answer"])
        # print(text_input)
        deceptive_document = self.model(text_input)[0]["generated_text"]
        deceptive_document = self.filter_doc(deceptive_document)
        row["wrong_document"] = deceptive_document
        # return deceptive_document
        return row


    def generate(self, df):
        df =  multi_process_map(df, self.generate_deceptive_doc, num_proc=64)
        df["docs"] = df.apply(lambda x: [{"text": x["wrong_document"], "title":""}], axis=1)
        return df