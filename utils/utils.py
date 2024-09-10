import json
import pandas as pd
import re
import string
import os
OPENAI_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o", "ft:gpt-4o-2024-08-06:duke-university:faithful-93:A3S28crx"]
def read_jsonl(file_path, return_df=True):
    # turn jsonl to dataframe
    data = []
    print(f"Reading {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    if return_df:
        return pd.DataFrame(data)
    else:
        return data

def save_jsonl(data, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
        print(f"Created directory {os.path.dirname(file_path)}")

    # turn dataframe to jsonl
    with open(file_path, "w") as f:
        # if directory does not exist, create it
        # decide if data is a dataframe or a list of dictionaries
        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
        else:
            for item in data:
                f.write(json.dumps(item) + "\n")

    print(f"Saved to {file_path}")


def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def get_file_contents(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as f:
        content = f.read()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_file_contents_as_list(file_path, encoding='utf-8', ignore_blanks=True):
    contents = get_file_contents(file_path, encoding=encoding)
    lines = contents.split('\n')
    lines = [line for line in lines if line != ''] if ignore_blanks else lines
    return lines


def extract_source_reliability(eval_file):
    if "sr:" not in eval_file:
        return None
    # Regular expression pattern to find "sr:" followed by a number (integer or decimal)
    pattern = r"sr:(\d+(\.\d+)?)"

    # Search the input string for the pattern
    match = re.search(pattern, eval_file)

    # If a match is found, return the matched number as a float
    if match:
        return float(match.group(1))
    else:
        return None


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


