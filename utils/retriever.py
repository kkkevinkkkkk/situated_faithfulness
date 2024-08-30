from transformers import DPRReader, DPRReaderTokenizer
import requests
from bs4 import BeautifulSoup
import torch
import numpy as np


class Retriever:
    def __init__(self):
        self.tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        self.model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base").to("cuda")
        self.model.eval()

    def retrieve(self, question, urls, k=5):
        retrieved_texts = []
        # if url is not string, return empty list
        if not isinstance(urls, str):
            return retrieved_texts

        if isinstance(urls, str):
            urls = urls.split("\n")
        for url in urls:
            # if get error, continue
            try:
                response = requests.get(url, timeout=10)
            except:
                print(f"error in getting url {url} with question {question}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            # Retrieve paragraphs
            paragraphs = soup.find_all('p')
            text_paragraphs = [p.get_text() for p in paragraphs]
            paragraphs = self.retrieve_paragraph(text_paragraphs, question, k)
            retrieved_texts.extend(paragraphs)

        return retrieved_texts

    def retrieve_answer(self, answer, question):
        encoded_inputs = self.tokenizer(
            questions=question,
            titles=[""],
            texts=[answer],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded_inputs = {k: v.to("cuda") for k, v in encoded_inputs.items()}
        outputs = self.model(**encoded_inputs)
        start_idx, end_idx = outputs.start_logits.argmax(dim=1)[0].item(), outputs.end_logits.argmax(dim=1)[0].item()
        answer = self.tokenizer.decode(encoded_inputs["input_ids"][0][start_idx:end_idx+1])
        return answer


    def retrieve_paragraph(self, text_paragraphs, question, k=5, batch_size=32):
        # get the text from the paragraphs, remove the text is text is \n or empty
        text_paragraphs = [p for p in text_paragraphs if p.strip() != ""]

        batched_paragraphs = [text_paragraphs[i:i + batch_size] for i in range(0, len(text_paragraphs), batch_size)]
        relevant_logits = []
        for batch in batched_paragraphs:
            encoded_inputs = self.tokenizer(
                questions=question,
                titles=[""] * len(batch),
                texts=batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            # Move to cuda
            encoded_inputs = {k: v.to("cuda") for k, v in encoded_inputs.items()}
            outputs = self.model(**encoded_inputs)
            # append the relevant logits
            relevant_logits.extend(outputs.relevance_logits.to("cpu").tolist())
            del outputs

            # delete model outputs from cuda
        # select the top k relevant paragraphs
        relevant_logits = np.array(relevant_logits).flatten()
        # get the relevant softmax probabilities
        relevant_probs = np.exp(relevant_logits) / np.sum(np.exp(relevant_logits))
        top_k_indices = np.argsort(relevant_logits)[::-1][:k]
        retrieved_texts = [text_paragraphs[i] for i in top_k_indices]

        return retrieved_texts


    def retrieve_table(self, soup, question, k=5):
        tables = soup.find_all('table')
        text_tables = []
        for table in tables:
            rows = table.find_all('tr')
            text_table = []
            for row in rows:
                cells = row.find_all(['th', 'td'])
                row_data = [cell.get_text().strip() for cell in cells]
                row_text = "    ".join(row_data)
                text_table.append(row_text)
            text_table = "\n".join(text_table)
            text_tables.append(text_table)
        encoded_inputs = self.tokenizer(
            questions=question,
            titles=[""] * len(text_tables),
            texts=text_tables,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt", )
        outputs = self.model(**encoded_inputs)
        retrieved_ids = torch.topk(outputs.relevance_logits, k).indices.to("cpu").tolist()
        retrieved_texts = [text_tables[int(i)] for i in retrieved_ids]
        return retrieved_texts

