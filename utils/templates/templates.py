# TEMPLATE_SYNTHESIZE_DOCUMENT = '''Given a question, a document supports this question, and the ground-truth answer to this document. Your job is to modify the document, so that the document will mislead the model to give a wrong answer to the question.
#
# To create a document that misleads the model into providing the wrong answer, consider incorporating details, facts, or narratives that support the wrong answer while maintaining the original document's style and structure.
#
# Question: {question}
#
# Document: {document}
#
# Answer: {answer}
#
# Assume the answer is "{wrong answer}" instead of "{answer}".
#
# Incorporate plausible but incorrect data points, expert opinions, or historical events that lend credibility to the wrong answer. If possible, introduce minor contradictions or ambiguities that might cause confusion or misinterpretation, leading the reader (or model) to the wrong conclusion.
#
# Ensure that the modified document subtly leads to the wrong conclusion without appearing blatantly incorrect or inconsistent with the document's overall context and tone.
# '''


# TEMPLATE_SYNTHESIZE_DOCUMENT = '''Your goal is to revise a document so it becomes a misleading source of information, leading readers (or models) to conclude with an incorrect answer when presented with a specific question. This manipulation should be subtle, weaving in plausible yet false details, expert opinions, or events that give credence to the incorrect answer. The challenge lies in altering the document without disrupting its original style, structure, and semblance of credibility.
# Question: {question}
#
# Document: {document}
#
# Answer: {answer}
#
# Assume the answer is "{wrong answer}" instead of "{answer}".
#
# Introduce facts, data points, or narratives that logically support the wrong answer. These additions should be believable and relevant to the document's topic. Despite the misleading modifications, the document should remain internally consistent, preserving the flow and coherence of the original text.
# '''


# TEMPLATE_SYNTHESIZE_DOCUMENT = '''Given a question, a document supports this question, and the ground-truth answer to this document. Your job is to modify the document, so that the document will mislead the model to give a wrong answer to the question.
#
# Question: {question}
#
# Document: {document}
#
# Answer: {answer}
#
# Assume the answer is "{wrong answer}" instead of "{answer}".
#
# Introduce facts, data points, or narratives that logically support the wrong answer. These additions should be believable and relevant to the document's topic. Despite the misleading modifications, the document should remain internally consistent, preserving the flow and coherence of the original text.
# '''

TEMPLATE_SYNTHESIZE_DECEPTIVE_DOCUMENT = '''Given a question, a document supports this question, and the ground-truth answer to this document. Your job is to modify the document so that the document will mislead the model to give a wrong answer to the question.

Question: {question}

Document: {document}

Answer: {answer}

Assume the answer is {deceptive_answer} instead of {answer}, and make up a new document with the same style. Incorporate plausible but incorrect data points, expert opinions, or historical events that lend credibility to the wrong answer.

Please only provide the revised document. There is no need to include the question, original document, correct answer, or any other additional information..
'''
TEMPLATE_GENERATE_MULTIPLE_CHOICE = '''You will be given a question and the ground-truth answer to it.  Your task is to generate ten potential answer candidates, making sure they are closely related yet distinct enough to test the student's depth of knowledge and attention to detail. Incorporate a mix of specific details, common misconceptions, and a few options that, while plausible, can be eliminated with careful thought or deeper knowledge. 

Question: {question}

Correct Answer: {answer}

Structure your answers in the format "<Answer 1>, <Answer 2>, ..."  and make sure the correct answer is the first.
'''

TEMPLATE_QA_TO_STATEMENT = '''Your task is to convert a question and an answer into a statement.

Here are some examples:

Question: Who wrote "Romeo and Juliet"?
Answer: William Shakespeare 
Statement: William Shakespeare wrote "Romeo and Juliet".

Question: What is the tallest building in the world?
Answer: Burj Khalifa
Statement: Burj Khalifa is the tallest building in the world.

Question: What is the capital of France?
Answer: Paris
Statement: Paris is the capital of France.

Now it's your turn. 
Question: {question}
Answer: {answer}
Statement:
'''

TEMPLATE_GENERATE_SUPPORT='''You will be given a question and an answer. Your goal is to craft a Wikipedia-style document that indirectly leads to the answer through reasoned inference.  To achieve that, you need to come up with a few plausible alternative answers that are related to the main topic but are not the correct answer. 

Then collect detailed information not only about the core answer but also about the alternative answers. This background should include historical context, scientific principles, cultural significance, or any other relevant data that can enrich the reader's understanding of the topic.

For each potential answer, including the correct one, provide a mix of direct information and indirect clues. Direct information establishes the relevance of each potential answer, while indirect clues hint at why it may or may not be the correct conclusion.

Finally, organize the document so that information flows in a logical but non-linear manner. Introduce facts and narratives related to the alternative answers in a way that initially obscures the path to the correct conclusion. This misdirection encourages readers to consider all possibilities before arriving at the correct inference.

Question: {question}

Answer: {answer}

Now provide your document in a paragraph.'''


TEMPLATE_LLAMA2_CHAT = '''<s>[INST] {task_instruction} [/INST]'''

# TEMPLATE_EXTRACT_ANSWER = '''Your task is to summarize the model's answer to a question into a brief response. The objective is to distill the essence of the model's reply into a concise phrase. This phrase should directly reflect the model's response, not the ground truth, ensuring it is as succinct as possible. You must preserve the integrity of the model's original answer, adapting it to match the concise format of the provided correct answer without changing its meaning or introducing new information.
#
# Here are some examples:
# Example 1:
# Question: What is the tallest building in the world?
# Ground Truth Answer: Burj Khalifa
# Model Response: The tallest building in the world is Taipei 101.
# Taipei 101
#
# Example 2:
# Question: Which actress was voted Miss Greenwich Village in 1942?
# Ground Truth Answer: Lauren Bacall
# Model Response:   The actress who was voted Miss Greenwich Village in 1942 was Marilyn Monroe.
# Marilyn Monroe
#
# Question: {question}
# Ground Truth Answer: {ground_truth}
# Model Response: {model_response}'''

TEMPLATE_EXTRACT_ANSWER = '''Your task is to distill the model's reply to a question into a concise and accurate response. Your goal is to capture the essence of the model's answer in a brief statement that directly mirrors the model's reply, without altering its original meaning or adding new information. To do this effectively, you must:\n\n1. Understand the type of answer expected (e.g., city, year, name) based on the question and the provided correct answer.\n2. Extract and condense the model's response to match the expected answer type, ensuring your summary faithfully reflects the model's original statement.\n3. Avoid introducing biases or assumptions not present in the model's reply, maintaining the integrity of the original response.\n\nYour summary should be clear and to the point, accurately reflecting the model's provided answer.

Here are some examples:
Example 1:
Question: What is the tallest building in the world?
Ground Truth Answer: Burj Khalifa
Model Response: The tallest building in the world is Taipei 101.
Taipei 101

Example 2:
Question: Which actress was voted Miss Greenwich Village in 1942?
Ground Truth Answer: Lauren Bacall
Model Response:   The actress who was voted Miss Greenwich Village in 1942 was Marilyn Monroe.
Marilyn Monroe

Question: {question}
Ground Truth Answer: {ground_truth}
Model Response: {model_response}'''

TEMPLATE_EXTRACT_ANSWER_0_SHOT = '''Your task is to distill the model's reply to a question into a concise and accurate response. Your goal is to capture the essence of the model's answer in a brief statement that directly mirrors the model's reply, without altering its original meaning or adding new information. To do this effectively, you must:\n\n1. Understand the type of answer expected (e.g., city, year, name) based on the question and the provided correct answer.\n2. Extract and condense the model's response to match the expected answer type, ensuring your summary faithfully reflects the model's original statement.\n3. Avoid introducing biases or assumptions not present in the model's reply, maintaining the integrity of the original response.\n\nYour summary should be clear and to the point, accurately reflecting the model's provided answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Model Response: {model_response}'''

TEMPLATE_EXTRACT_SHORT_DOC = '''Given a document and a question, identify and extract the minimal number of sentences needed to directly answer the question, ideally limiting to one sentence if possible. Replace any pronouns in these sentences with the specific entities they refer to, ensuring clarity and coherence. Only provide your extraction as your response.

Question: {question}

Document: {document}'''

TEMPLATE_TURN_MC = '''You will receive a question along with a correct answer and an incorrect answer from a document. Your task is to convert this question into a multiple-choice format. You should provide four answer choices: one correct answer, one incorrect document answer, and two other plausible but incorrect answers.

You must return the answer choices, the correct answer choice, and the wrong document answer choice. 

Example 1:

Question: What is the capital of France?
Correct Answer: Paris
Incorrect Document Answer: Lyon

Answer Choices:
A) London
B) Paris
C) Berlin
D) Lyon

Correct Answer: B) Paris
Incorrect Document Answer: D) Lyon

Example 2:

Question: Who wrote "Romeo and Juliet"?
Correct Answer: William Shakespeare
Incorrect Document Answer: Charles Dickens

Answer Choices:
A) William Shakespeare
B) Bram Stoker
C) Charles Dickens
D) Jane Austen

Correct Answer: A) William Shakespeare
Incorrect Document Answer: C) Charles Dickens

Example 3:

Question: What is the tallest building in the world?
Correct Answer: Burj Khalifa
Incorrect Document Answer: Taipei 101

Answer Choices:
A) Empire State Building
B) Taipei 101
C) Willis Tower
D) Burj Khalifa

Correct Answer: D) Burj Khalifa
Incorrect Document Answer: B) Taipei 101

Now it's your turn.

Question: {question}
Correct Answer: {correct_answer}
Incorrect Document Answer: {incorrect_document_answer}

'''


TEMPLATES = {
    "llama2_chat": TEMPLATE_LLAMA2_CHAT,
    "synthesize_deceptive_document": TEMPLATE_SYNTHESIZE_DECEPTIVE_DOCUMENT,
    "qa_to_statement": TEMPLATE_QA_TO_STATEMENT,
    "generate_support": TEMPLATE_GENERATE_SUPPORT,
    "generate_multiple_choice": TEMPLATE_GENERATE_MULTIPLE_CHOICE,
    "extract_answer": TEMPLATE_EXTRACT_ANSWER,
    "extract_answer_0_shot": TEMPLATE_EXTRACT_ANSWER_0_SHOT,
    "extract_short_doc": TEMPLATE_EXTRACT_SHORT_DOC,
    "turn_mc": TEMPLATE_TURN_MC
}




