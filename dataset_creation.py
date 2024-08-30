from utils import Retriever, NLIModel, read_jsonl, save_jsonl
import requests
from bs4 import BeautifulSoup
from time import sleep
import os
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
from IPython.display import display, Image as IPImage
from tqdm import tqdm
class FactCheckQACreator:
    def __init__(self):
        self.retriever = Retriever()
        self.nli_model = NLIModel()
        self.image_cache_dir = '/usr/xtmp/yh386/faithfulness/cache/images'
        self.session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        self.session.headers = headers

    def retry_request(self, url, max_retries=3, retry_delay=5.0):

        current_delay = retry_delay
        response = None
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                return response
            except requests.RequestException as e:
                # print(f"Attempt {attempt + 1} failed: {e}")
                # if invalid url, break the loop
                if str(e).startswith("Invalid URL"):
                    break
                # if error of the RequestException is 404, break the loop
                if response is not None:
                    if response.status_code == 404:
                        break
                if attempt < max_retries - 1:
                    sleep(current_delay)  # Wait before retrying
                    current_delay += retry_delay  # backoff
                else:
                    print(f"All attempts failed for {url}")
        return None

    def find_archived_text(self, row):
        archived_url = row["archived_url"]
        fact_claim = row["claim_text"]
        response = self.retry_request(archived_url, max_retries=5)
        if not response:
            return []

        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        paragraphs = soup.find_all('p')
        text_paragraphs = [p.get_text() for p in paragraphs]
        return text_paragraphs

    def find_archived_text_from_images(self, row):
        archived_url = row["archived_url"]
        original_id = row["original_id"]

        response = self.retry_request(archived_url, max_retries=3)
        if not response:
            return []

        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        images = soup.find_all('img')
        image_urls = [img['src'] for img in images if 'src' in img.attrs]
        saved_images = []
        # filter the repetition in the image urls
        image_urls = list(set(image_urls))

        for i, url in enumerate(image_urls):
            if url.endswith('gif') or url.startswith('data:image') or url.startswith('/web/2021'):
                continue
            if not url.startswith('http') and not url.startswith('//'):
                if len(url.split('/')) > 1:
                    if url.split('/')[1] == archived_url.split('/')[-1]:
                        url = "/".join(url.split('/')[2:])

                if url.startswith('/'):
                    url = f"{archived_url}{url}"


            response = self.retry_request(url, max_retries=3)
            if response:
                image_cache_path = os.path.join(self.image_cache_dir, f"fact_{original_id}")
                # create the cache directory if it doesn't exist
                os.makedirs(image_cache_path, exist_ok=True)
                image_path = os.path.join(image_cache_path, f"image_{i}.jpg")
                # save the image to the cache directory
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                    # print(f"Downloaded image {i} of fact {original_id} to {image_path}")
                saved_images.append(image_path)
        text_paragraphs = []
        for image_path in saved_images:
            try:
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                text_paragraphs.append(text)
            except Exception as e:
                print(f"Error in reading image {image_path}: {e}")
            # image = Image.open(image_path)
            # text = pytesseract.image_to_string(image)
            # text_paragraphs.append(text)
        return text_paragraphs

    def filter_paragraphs(self, paragraphs):
        paragraphs = [p.strip() for p in paragraphs if len(p.split()) > 10]
        filtered_parts = ["Wayback Machine", "Search the history of over", "Capture a web page as it appears now", "keep fighting for all libraries" ]
        filtered_paragraphs = []
        for paragraph in paragraphs:
            if any(part.lower() in paragraph.lower() for part in filtered_parts):
                continue
            filtered_paragraphs.append(paragraph)
        return filtered_paragraphs

    def find_supporting_doc(self, text_paragraphs, claim_text):
        text_paragraphs = self.filter_paragraphs(text_paragraphs)
        all_paragraphs = text_paragraphs
        retrieved_paragraphs = self.retriever.retrieve_paragraph(text_paragraphs, claim_text, k=5,
                                                                 batch_size=32)
        supporting_paragraphs = []
        for paragraph in retrieved_paragraphs:
            is_evidence = self.nli_model.run(paragraph, claim_text)
            if is_evidence:
                supporting_paragraphs.append(paragraph)
        if len(supporting_paragraphs) == 0:
            return retrieved_paragraphs, False, all_paragraphs

        return supporting_paragraphs, True, all_paragraphs

    def find_archived_doc(self, row):
        if row["archived_url"] is None:
            return [], False, []
        text_paragraphs = self.find_archived_text(row)
        supporting_paragraphs, find_nli_paragraphs, all_paragraphs = self.find_supporting_doc(text_paragraphs, row["claim_text"])
        if not find_nli_paragraphs:
            text_paragraphs = self.find_archived_text_from_images(row)
            supporting_paragraphs_images, find_nli_paragraphs, all_paragraphs_images = self.find_supporting_doc(text_paragraphs, row["claim_text"])
            supporting_paragraphs.extend(supporting_paragraphs_images)
            all_paragraphs.extend(all_paragraphs_images)
        return supporting_paragraphs, find_nli_paragraphs, all_paragraphs


if __name__ == '__main__':
    creator = FactCheckQACreator()

    df_fcqa = read_jsonl('/usr/xtmp/yh386/datasets/FactCheckQA/FactCheckQA_v1-1.jsonl')
    archived_paragraphs = []
    find_nlis = []
    all_paragraphs = []
    for i, row in tqdm(df_fcqa.iterrows()):
        # if i<15667:
        #     continue
        paragraphs, find_nli_paragraphs, local_all_paragraphs = creator.find_archived_doc(row)
        archived_paragraphs.append(paragraphs)
        find_nlis.append(find_nli_paragraphs)
        all_paragraphs.append(local_all_paragraphs)

    df_fcqa['archived_docs'] = archived_paragraphs
    df_fcqa['find_nli'] = find_nlis
    df_fcqa['all_docs'] = all_paragraphs
    save_jsonl(df_fcqa, '/usr/xtmp/yh386/datasets/FactCheckQA/FactCheckQA_v1-2.jsonl')
