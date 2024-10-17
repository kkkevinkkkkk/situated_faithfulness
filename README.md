# Situated Faitfhulness
## Enhancing Large Language Models' Situated Faithfulness to External Contexts

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)]()

This is the repository of source code for our paper "Enhancing Large Language Models' Situated Faithfulness to External Contexts". The paper can be found at [arXiv]().

## Table of Contents
   - [Install](#install)
   - [Data](#data)
   - [Run](#run)



## Install
The first things you need to do are cloning this repository and installing its
dependencies:

```sh
git clone hhttps://github.com/kkkevinkkkkk/situated_faithfulness.git
cd situated_faithfulness
pip install -r requirements.txt
```

## Data
Situated Faithfulness benchmark includes RedditQA, TriviaQA, FreshQA, NaturalQA, ConflictQA(Only include PopQA), ClashEval. 
The Evaluation data is available on [huggingface](https://huggingface.co/datasets/kkkevinkkk/SituatedFaithfulnessEval). The training data for CR-DPO could be downloaded [here](https://huggingface.co/datasets/kkkevinkkk/SituatedFaithfulnessSupplement).
Each dataset includes columns: `question`, `answers`(a list of reference answers), `correct_doc`, `wrong_doc`, `correct_answer`(the answer from the correct document), `wrong_answer`(the answer from the wrong document), and other dataset specific columns. Each dataset has a test split and a dev split.

You could download the data using the jupyter notebook [download_dataset.ipynb](download_dataset.ipynb)



## Run
Configure files of running the models are in [configures/](configures). You could change the settings by changing the corresponding `.yml` file.

There are demos for following methods:
1. `Closed-book`: get model's internal answer ([configures/demo_cb.yml](configures/demo_cb.yml) )
2. `Direct Input Augmentation`: directly ask the model to utilize the context to get the answer ([configures/demo_dia.yml](configures/demo_dia.yml))
3. `Complete Faithful`:  ask model to be complete faithful to the context ([configures/demo_complete.yml](configures/demo_complete.yml))
4. `Implicit Self-guided Reasoning`: ask model to do implicit self-guided reasoning ([configures/demo_iscr.yml](configures/demo_iscr.yml))
5. `Explicit Self-guided Reasoning`: ask model to do explicit self-guided reasoning ([configures/demo_escr.yml](configures/demo_escr.yml))
You could run the demo by:
```sh
python run.py --config_path configures/demo_cb.yml
```


You could change the task by changing the `dataset_name` and `eval_file` in the configure file.
Supported tasks are `redditqa`, `triviaqa`, `freshqa`, `naturalqa`, `conflictqa` (popqa), `clash_eval`.
Models can be changed by modify the `model` in the configure file.  (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `meta-llama/Meta-Llama-3-8B-Instruct`)
> Note: when changing the model to llama3, the `multi_process` need to be set to False.



