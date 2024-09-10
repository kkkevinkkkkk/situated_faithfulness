from .evaluation_triviaqa import normalize_answer
import numpy as np
import re
from pipeline import LLM
from .utils import f1_score, metric_max_over_ground_truths

import datetime
import pytz
from utils import multi_process_map
current_date = datetime.datetime.now(
        pytz.timezone("America/Los_Angeles")
    ).strftime("%B %d, %Y")

prefix = (
    "Please evaluate the response to a question under relaxed evaluation, where"
    " hallucinations, outdated information, and ill-formed answers are allowed,"
    " as long as the primary answer is accurate. Please credit the response"
    " only if it provides a confident and definitive answer, or the correct"
    " answer can be obviously inferred from the response. The primary or final"
    " answer when standing alone must be accurate. Any additional information"
    " that is provided must not contradict the primary answer or reshape one's"
    " perception of it. For false-premise questions, the response must point"
    " out the presence of a false premise to receive credit. For answers that"
    " involve names of entities (e.g., people), complete names or commonly"
    " recognized names are expected. Regarding numerical answers, approximate"
    " numbers are generally not accepted unless explicitly included in the"
    " ground-truth answers. We accept ill-formed responses (including those in"
    " a non-English language), as well as hallucinated or outdated information"
    " that does not significantly impact the primary answer."
)

demo_examples = [
    {
        "question": "How old is the world's oldest verified living person?",
        "correct_answers": [
            "117 years old",
            "117",
        ],
        "response": (
            f"As of today {current_date}, the most up-to-date and relevant"
            " information regarding this query is as follows. The oldest"
            " verified living person is Maria Branyas Morera, who was born"
            " on March 4, 1907, making her 117 years old."
        ),
        "comment": (
            "This is a valid question. The primary answer in the response"
            " (117 years old) is accurate, and all the provided information"
            " is accurate and up-to-date. Thus, the response is credited."
        ),
        "evaluation": "correct",
    },
    {
        "question": "When did the UK adopt the Euro?",
        "correct_answers": ["The United Kingdom has never adopted the Euro."],
        "response": (
            "The UK has never adopted the Euro as its official currency. The"
            " country has retained the British pound sterling (GBP) as its"
            " currency."
        ),
        "comment": (
            "The question contains a false premise. The response debunks the"
            " question's false premise (i.e., it points out that the UK has"
            " never adopted the Euro), and all the provided information is"
            " accurate and up-to-date. Thus, the response is credited."
        ),
        "evaluation": "correct",
    },
    {
        "question": (
            "How much longer will Brittney Griner spend in the Russian prison?"
        ),
        "correct_answers": [
            "She was released in December 2022 as part of a prisoner swap."
        ],
        "response": (
            "I'm sorry, but I have no information to suggest that Brittney"
            " Griner is currently in a Russian prison. As far as I know, she"
            " is a professional basketball player who currently plays for the"
            " Phoenix Mercury in the Women's National Basketball Association"
            " (WNBA). If you have any more information or context, please let"
            " me know and I'll do my best to help."
        ),
        "comment": (
            "The question contains a false premise. The response fails to"
            " debunk the question's false premise by not explicitly stating"
            " that Brittney Griner was already released. Additionally, the"
            " response lacks a confident and definitive answer. Thus, the"
            " response is not credited."
        ),
        "evaluation": "incorrect",
    },
    {
        "question": (
            "Which language has the third largest population by number of"
            " native speakers?"
        ),
        "correct_answers": ["English"],
        "response": "1. Mandarin 2. Spanish 3. English",
        "comment": (
            "This is a valid question. Even though the response doesn't contain"
            " an explicit statement, the correct answer (English) can still be"
            " inferred from the response. Thus, the response is credited."
        ),
        "evaluation": "correct",
    },
    {
        "question": (
            "Has Virginia Woolf's novel about the Ramsay family entered the"
            " public domain in the United States?"
        ),
        "correct_answers": ["Yes"],
        "response": (
            "1927's To the Lighthouse is in the public domain in the United"
            " States. The final answer is To the Lighthouse."
        ),
        "comment": (
            "This is a valid question. While the primary information in the"
            " response (1927's To the Lighthouse is in the public domain in"
            " the United States) is accurate, the final answer when standing"
            ' alone is inaccurate (it should be "Yes" instead of "To the'
            ' Lighthouse"). Thus, the response is not credited.'
        ),
        "evaluation": "incorrect",
    },
    {
        "question": "Is Tesla's stock price above $250 a share?",
        "correct_answers": ["No"],
        "response": "No, it isn't. The stock price is currently at $257.",
        "comment": (
            "This is a valid question. While the primary answer in the"
            " response (No) is accurate, the additional information"
            " contradicts the primary answer ($257 is above $250). Thus,"
            " the response is not credited."
        ),
        "evaluation": "incorrect",
    },
    {
        "question": "When did Lionel Messi win his second FIFA World Cup?",
        "correct_answers": [
            "Messi has not won his second FIFA World Cup.",
            "Messi has won only one FIFA World Cup.",
        ],
        "response": (
            "This question does not contain a valid premise. Lionel Messi has"
            " never won a FIFA World Cup."
        ),
        "comment": (
            "The question contains a false premise. While the response"
            " debunks the question's false premise (i.e., it suggests that"
            " Messi has not won his second FIFA World Cup), it inaccurately"
            " shapes one's knowledge perception of the primary answer, as"
            " Messi has already won one FIFA World Cup. Thus, the response"
            " is not credited."
        ),
        "evaluation": "incorrect",
    },
    {
        "question": (
            "How many times has Jeremy Corbyn served as Prime Minister of"
            " the UK?"
        ),
        "correct_answers": [
            "Corbyn has never served as Prime Minister of the UK."
        ],
        "response": "Zero",
        "comment": (
            "The question contains a false premise. The response lacks a"
            " rebuttal, which is always required for false-premise questions."
            " Thus, the response is not credited."
        ),
        "evaluation": "incorrect",
    },
    {
        "question": "Who founded Amazon?",
        "correct_answers": ["Jeff Bezos", "Jeffrey Preston Bezos"],
        "response": "100% of people said Jeff.",
        "comment": (
            "This is a valid question. Given that the question involves a"
            " person's name, a complete name or commonly recognized name is"
            ' expected ("Jeff Bezos" instead of "Jeff"). Thus, the response'
            " is not credited."
        ),
        "evaluation": "incorrect",
    },
    {
        "question": "How many books has Colleen Hoover published?",
        "correct_answers": ["26 books", "26"],
        "response": (
            f"As of today {current_date}, the most up-to-date and relevant"
            " information regarding this query is as follows. Colleen Hoover"
            " has published over 20 books and novellas."
        ),
        "comment": (
            "This is a valid question. For a numerical answer, an approximate"
            " value (over 20 books) is generally not accepted unless"
            " explicitly included in the correct answers. Thus, the response"
            " is not credited."
        ),
        "evaluation": "incorrect",
    },
    {
        "question": "What is the largest model size of GPT-3?",
        "correct_answers": [
            "175 billion parameters",
            "175B parameters",
            "175 billion",
            "175B",
        ],
        "response": (
            "175 billion parameters. It is a 175 billion parameter, 1.37TB,"
            " 137B token, 137B wordpiece, 137B BPE, 137B byte pair encoding,"
            " 137B BPE token, 137B BPE wordpiece, 137B BPE token, 137B BPE"
            " wordpiece"
        ),
        "comment": (
            "This is a valid question. Even though the response is ill-formed"
            " and contains repetitions, the primary answer in the response (175"
            " billion parameters) is still accurate. Thus, the response is"
            " credited."
        ),
        "evaluation": "correct",
    },
    {
        "question": "How many member nations exist in the United Nations?",
        "correct_answers": [
            "193",
            "193 members",
            "193 member states",
            "193 sovereign states",
        ],
        "response": (
            "As of my last update in April 2023, there were 193"
            " member nations in the United Nations. Please note that this"
            " information may have changed since then, so it is advisable to"
            " check the latest information from the United Nations or other"
            " reliable sources."
        ),
        "comment": (
            "This is a valid question. Even though the response fails to"
            " provide a current answer and instead relies on outdated "
            " information without confirming its continued accuracy, the"
            " primary answer in the response (193 member nations) is still"
            " accurate. Thus, the response is credited."
        ),
        "evaluation": "correct",
    },
    {
        "question": (
            "Who won the biggest single-tournament payday in tennis history?"
        ),
        "correct_answers": ["Novak Djokovic"],
        "response": (
            "The biggest single-tournament payday in tennis history was won"
            " by Novak Djokovic at the 2021 Australian Open. He received a"
            " prize money of AUD 2.75 million (approximately USD 2.1 million)"
            " for winning the men's singles title."
        ),
        "comment": (
            "This is a valid question. Even though the additional provided"
            " information is outdated or hallucinated (by winning the 2022 ATP"
            " Finals title, Djokovic earned a record $4,740,300, the highest"
            " payout for a tournament winner in tennis), the primary answer in"
            " the response (Novak Djokovic) is still accurate. Thus, the"
            " response is credited."
        ),
        "evaluation": "correct",
    },
    {
        "question": (
            "Who was the first football player to score one hundred"
            " international goals?"
        ),
        "correct_answers": ["Ali Daei"],
        "response": (
            "As of my last update in April 2023, Ali Daei, a"
            " former Iranian footballer, was the first player to score 100"
            " international goals."
        ),
        "comment": (
            "This is a valid question. The primary answer in the response"
            " (Ali Daei) is accurate. Although the knowledge cutoff date is"
            " in 2021, this fact is timeless because Ali Daei was the initial"
            " achiever of this milestone, and he will always remain the first)."
            " Thus, the response is credited."
        ),
        "evaluation": "correct",
    },
    {
        "question": (
            "Who is the latest winner of the Formula 1 world championship?"
        ),
        "correct_answers": ["Max Verstappen", "Max Emilian Verstappen"],
        "response": (
            "Max Verstappen won the 2023 Formula 1 world championship. He is"
            " a Dutch racing driver who competes in Formula One for Red Bull"
            " Racing. He is the son of former Formula One driver Jos"
            " Verstappen. Verstappen started his racing career in karting at"
            " the age of seven. He won the 2013 European Karting Championship"
            " and the 2013 World Karting Championship. In 2014, he moved to"
            " single-seater racing, competing in the European Formula Three"
            " Championship. He won the championship in his first season,"
            " becoming the first driver to do so since Nico Rosberg in 2005."
            " In 2015, Verstappen moved to Formula One, driving for Toro"
            " Rosso. He became the youngest driver to compete in Formula One"
            " at the age of 17. He scored his first points in Formula One at"
            " the 2015 Hungarian GrandPrix. In 2016, Verstappen moved to Red"
            " Bull Racing. He won his first race at the 2016 Spanish Grand"
            " Prix. He became the youngest driver to win a race in Formula"
            " One at theage of 18. Verstappen finished the 2016 season in"
            " third place in the drivers' championship. In 2017, Verstappen"
            " won four races and finished the season in second place in the"
            " drivers' championship. In 2018, Verstappen won seven races and"
            " finished the season in second place in the drivers'"
            " championship. In 2019, Verstappen won nine races and finished"
            " the season in first place in the drivers' championship. He is"
            " the first Dutch driver to win the Formula One world"
            " championship."
        ),
        "comment": (
            "This is a valid question. Even though the response contains"
            " several instances of hallucinated information (e.g., Max"
            " Verstappen did not win the Formula Three European Championship in"
            " 2014), the primary answer in the response (Max Verstappen) is"
            " still accurate. Thus, the response is credited."
        ),
        "evaluation": "correct",
    },
]

demo_questions = [ex["question"] for ex in demo_examples]

demo_evaluation_template = (
    "\ncorrect answer(s): {correct_answers}"
    "\nresponse: {response}"
    "\ncomment: {comment}"
    "\nevaluation: {evaluation}"
)
evaluation_template = (
    "\ncorrect answer(s): {correct_answers}"
    "\nresponse: {response}"
    "\ncomment: "
)

demo_evaluations = []
for ex in demo_examples:
    demo_evaluation = demo_evaluation_template.format(
        question=ex["question"],
        correct_answers=' | '.join(ex["correct_answers"]),
        response=ex["response"],
        comment=ex["comment"],
        evaluation=ex["evaluation"],
    )
    demo_evaluations.append(demo_evaluation)


def extract_ratings(response):
    evaluation = None
    for line in response.split('\n'):
        if 'evaluation: ' in line:
            evaluation = line.split(' ')[-1]
            if evaluation not in ['correct', 'incorrect']:
                return False, {'rating': None}
            if evaluation == 'incorrect':
                evaluation = 'FALSE'
            else:
                evaluation = 'TRUE'
    if evaluation is None:
        if 'Thus, the response is credited.' in response:
            evaluation = 'TRUE'
        elif 'Thus, the response is not credited.' in response:
            evaluation = 'FALSE'
        else:
            return False, {'rating': None}
    return True, {'rating': evaluation}


def evaluate_naturalqa_answer_f1(row):
    question = row["question"]
    correct_answers = row["answers"]
    response = row["generated_text"]
    response = get_answer(response)
    f1 = metric_max_over_ground_truths(
        f1_score, response, correct_answers)
    return float(f1)


def evaluate_naturalqa_answer_gpt4(row):
    temperature = 0.0
    max_tokens = 256
    chat_completions = True
    model = "gpt-4o-mini"
    model = "gpt-4o"
    pipe = LLM(model_name=model)

    question = row["question"]
    correct_answers = row["answers"]
    response = row["generated_text"]

    # Generate prompts for demo examples
    demo_prompts = []
    for q, e in zip(demo_questions, demo_evaluations):
        demo_prompts.append(f'\n\n\nquestion: {q}{e}')

    fresheval_demo = ''.join(demo_prompts).strip()

    evaluation = evaluation_template.format(
        correct_answers=' | '.join(correct_answers),
        response=response,
    )
    fresheval_question = f'\n\n\nquestion: {question}{evaluation}'

    fresh_eval = prefix + '\n\n\n' + fresheval_demo + fresheval_question
    is_valid_eval = False
    max_eval_attempts = 3
    current_attempt = 0
    while not is_valid_eval and current_attempt < max_eval_attempts:
        result = pipe(fresh_eval, temperature=temperature, max_tokens=max_tokens)[0]['output']
        is_valid_eval, eval = extract_ratings(result)
        current_attempt += 1

    is_correct = eval['rating'] == 'TRUE'
    if not is_valid_eval:
        is_correct = False
        print(fresh_eval)

    return is_correct



# evaluate_naturalqa_answer = evaluate_naturalqa_answer_f1
evaluate_naturalqa_answer = evaluate_naturalqa_answer_gpt4
def get_answer(x):
    x = normalize_answer(x)
    # x = x.strip()
    return x


def evaluate_naturalqa(df):

    df = multi_process_map(df, evaluate_naturalqa_row)
    # df = df.apply(evaluate_naturalqa_row, axis=1)
    scores = df["relaxed_score"]
    total_scores = {"accuracy": scores.mean()}

    return total_scores, scores

def evaluate_naturalqa_row(row):
    row["relaxed_score"] = evaluate_naturalqa_answer(row)
    return row

def evaluate_naturalqa_expected_correctness(row):
    model_answer0 = row["generated_text"]
    other_answers = row["other_answers"]
    model_answers = [model_answer0] + other_answers
    scores = []
    for model_answer in model_answers:
        row["generated_text"] = model_answer
        score = evaluate_naturalqa_answer(row)
        scores.append(score)
    row["expected_correctness"] = np.mean(scores)
    row["generated_text"] = model_answer0
    # row["relaxed_score"] = scores[0]
    return row


