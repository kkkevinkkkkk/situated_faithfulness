import logging
import numpy as np
from utils.templates import TEMPLATES
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



DOC_PROMPT = "Document [{ID}](Title: {T}): {P}\n"

def make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=None):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = doc['text']
    if use_shorter is not None:
        text = doc[use_shorter]
    if "title" in doc:
        doc_prompt = doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))
    else:
        doc_prompt = doc_prompt.replace("{T}", "").replace("{P}", text).replace("{ID}", str(doc_id+1))

    if "document_confidence" in doc:
        if "{DC}" in doc_prompt:
            doc_prompt = doc_prompt.replace("{DC}", doc["document_confidence"])
    else:
        assert "{DC}" not in doc_prompt

    return doc_prompt

def get_shorter_text(eval_item, docs, ndoc, key):
    doc_list = []
    for item_id, item in enumerate(docs):
        if key not in item:
            if len(doc_list) == 0:
                # If there aren't any document, at least provide one (using full text)
                item[key] = item['text']
                doc_list.append(item)
            logger.warning(f"No {key} found in document. It could be this data do not contain {key} or previous documents are not relevant. This is document {item_id}. This question will only have {len(doc_list)} documents.")
            break
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
        doc_list.append(item)
        if len(doc_list) >= ndoc:
            break
    return doc_list


def make_demo(item, template,
              n_doc=None,
              doc_prompt=None,
              instruction=None,
              use_shorter=None,
              test=False):
    # For demo template
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    # qa dataset
    assert "question" in item
    if instruction is None or instruction == "":
        prompt = template.replace('{INST}\n\n', "")

        prompt = prompt.replace("{Q}", item['question'])
    else:
        prompt = template.replace("{INST}", instruction).replace("{Q}", item['question'])


    if "{D}" in prompt:
        if n_doc == 0:
            prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
        else:
            doc_list = get_shorter_text(item, item["docs"], n_doc, use_shorter) if use_shorter is not None else item["docs"][:n_doc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "") + answer
        # prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt


def make_head_prompt(prompt_data: dict,
                     n_shot: int = 0,
                     n_doc: int = 0,
                     n_doc_in_demo: int = 0,
                     fewer_doc_in_demo: bool = False,
                     no_doc_in_demo: bool = True,
                     use_shorter: str = "summary",
                     post_demo_instruction: str = "Now let's answer:\n\n"
                     ):
    train_ids = np.random.choice(len(prompt_data["demos"]), n_shot, replace=False)
    head_prompt = prompt_data["instruction"] + "\n\n"
    if n_shot == 0:
        return head_prompt

    if n_shot > 0:
        head_prompt += "Here are some examples:\n\n"

    for train_id in train_ids:
        train_item = prompt_data["demos"][train_id]
        n_doc = n_doc
        if no_doc_in_demo:
            n_doc = 0
        elif fewer_doc_in_demo:
            assert n_doc_in_demo is not None
            n_doc = n_doc_in_demo
        head_prompt += make_demo(
            train_item, template=prompt_data["demo_prompt"], n_doc=n_doc, doc_prompt=prompt_data["doc_prompt"],
            instruction=None, use_shorter=use_shorter
        )
        head_prompt += prompt_data["demo_sep"]

    head_prompt += post_demo_instruction
    return head_prompt




