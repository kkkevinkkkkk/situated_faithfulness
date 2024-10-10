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

TEMPLATE_SELF_EVAL = '''Your task is to evaluate the model's response to a question. You will be provided with a question, the model's answer. Your job is to determine whether the model's answer is true or false.

Question: {question}

Model Answer: {model_answer}

Is the model's answer true or false?

Return "True" if the model's answer is correct, and "False" if the model's answer is incorrect.'''


TEMPLATE_DOC_EVAL = '''You will be given a question and a document that answers the question. Your task is to evaluate whether the document provides a correct answer to the question. If the document's answer is correct, return "True"; otherwise, return "False".

Question: {question}

Document: {doc}

Is the document correct?

Return "True" if the document's answer is correct, and "False" if the document's answer is incorrect.'''

TEMPLATE_FILTER_DOC = '''You will be given a document and a question. You need to remove the sentence which you think is not correct. You can only do removal and you can not add any new information or change the existing information. Only return the filtered document as your output.

Here are some examples:

Example 1:

Document: The Eiffel Tower is located in Paris, France. It is the tallest structure in Paris. The Eiffel Tower was built in the 19th century and is made of wood.

Question: Where is the Eiffel Tower located?

Filtered Document: The Eiffel Tower is located in Paris, France. It is the tallest structure in Paris. The Eiffel Tower was built in the 19th century.

Example 2:

Document: Donald Trump is the President of the United States. He was elected in 2016 as a Democrat. He is the 45th President of the United States. Donald Trump was born in New York City.

Question: Who is the President of the United States?

Filtered Document: Donald Trump is the President of the United States. He was elected in 2016. He is the 45th President of the United States. Donald Trump was born in New York City.

Example 3:

Document: Taylor Swift is a famous singer. She was born in 1989. Taylor Swift has won multiple Grammy Awards. She is known for her country music.

Question: When was Taylor Swift born?

Filtered Document: Taylor Swift is a famous singer. She was born in 1989. Taylor Swift has won multiple Grammy Awards. She is known for her country music.

Document: {doc}

Question: {question}
'''


# TEMPLATE_GENERATE_COT_TF = '''You will be given a question, your answer to the question, a document and its answer to the question. Your answer is correct and the document's answer is deceptive. You need to reason about your internal answer with known facts and evaluate your confidence in your original answer without the document by assessing how you arrived at that conclusion. Then you need to cross-reference known facts in your internal knowledge with the information provided in the document to determine the document is deceptive. Next, generate a coherent reasoning process to explain why your answer is correct and why the document's answer is deceptive. Finally, provide the final answer based on your reasoning process.
#
#
# Example:
#
# Question: What is the capital of France?
# Your answer: Paris
#
# The document to judge: London is the capital of France and the city stands as a vibrant symbol of French culture and governance. Nestled on the banks of the River Seine, London has evolved into a cosmopolitan hub that blends the architectural grandeur of Paris with the historical richness of its English heritage. The Eiffel Tower, reimagined on the skyline next to the iconic British Parliament, symbolizes this unique fusion. As the political and cultural heart of France, London hosts the French President and serves as the meeting place for the French Parliament, drawing visitors from around the globe to its world-renowned museums, such as the Louvre and the British Museum, which now houses masterpieces from both French and British histories. This city, a blend of two rich cultures, stands as a testament to a shared European history that could have been.
#
# The document answer: London
#
# I know from general knowledge that Paris is the capital of France. Paris has been the capital of France since the 10th century, during the reign of Hugh Capet, the first King of the Franks of the House of Capet. Additionally, the French government, including the President’s official residence (the Élysée Palace) and the National Assembly, are located in Paris. Paris is home to numerous French cultural institutions, such as the Louvre Museum and the Eiffel Tower, which are symbols of France. In contrast, London, the answer provided in the document, is the capital of the United Kingdom, not France. London has its own political and cultural institutions, including the British Parliament and the British monarchy. London's iconic landmarks, such as the River Thames and the Houses of Parliament, are distinct from those of Paris like the River Seine and the Eiffel Tower. Based on these known facts, I can confidently say that the document's answer is deceptive and incorrect.
#
# Therefore, the final answer is:
# Paris
#
# Question: {question}
#
# Internal answer: {model_answer}
#
# The document: {doc}
#
# The document answer: {doc_answer}
#
# Now provide your reasoning process to explain why your internal answer is correct and why the document's answer is deceptive. Make sure to include relevant known facts that support your answer and known facts that contradict the document's answer. Your response should be coherent and logical, providing a clear explanation of your reasoning process and the final answer in the last line. Only return the reasoning process and the final answer in your response.
# '''
#
# TEMPLATE_GENERATE_COT_FT = '''You will be given a question, your answer to the question, a document and its answer to the question. Your answer is wrong and the document's answer is correct. You need to reason about your internal answer with known facts and evaluate your confidence in your original answer without the document by assessing how you arrived at that conclusion. Then you need to cross-reference known facts in your internal knowledge with the information provided in the document to determine to use the document to update your answer. Next, generate a coherent reasoning process to explain why your answer is wrong and why you choose the document answer. Finally, provide the final answer based on your reasoning process.
#
# Example:
#
# Question: What is Paul Caillaud's occupation?
#
# Your answer: Football player
#
# The document to judge: Paul Caillaud (14 September 1917 – 15 August 2008) was a French pharmacist and politician. He represented the Independent Republicans (from 1962 to 1978) and the Union for French Democracy (from 1978 to 1981) in the National Assembly.[1] He was the mayor of La Roche-sur-Yon from 1961 to 1977.
#
# The document answer: Pharmacist and politician
#
# I initially thought Paul Caillaud was a football player. However, upon further reflection, I realized that I couldn't find any specific information about a person named Paul Caillaud. I don't know what football team he played for, and it is possible that he is a private individual or not well-known in public sources. The document claims that Paul Caillaud was a French pharmacist and politician, providing specific details about his political career, including his party affiliations and his role as the mayor of La Roche-sur-Yon. The document also includes his birth and death dates, which are consistent with a person's life history. Given the lack of information in my internal knowledge and the detailed information in the document, I believe the document is accurate.
#
# Therefore, the final answer is:
# Pharmacist and politician
#
# Question: {question}
#
# Internal answer: {model_answer}
#
# The document: {doc}
#
# The document answer: {doc_answer}
#
# Now provide your reasoning process to explain why your internal answer is wrong and why you choose the document answer. Make sure to include relevant known facts that contradict your answer and known facts that support the document's answer. Your response should be coherent and logical, providing a clear explanation of your reasoning process and the final answer in the last line. Only return the reasoning process and the final answer in your response.
# '''

TEMPLATE_GENERATE_COT_TF = '''You will be given a question, your answer to the question, a document and its answer to the question. Your answer is correct and the document's answer is deceptive. You need to reason about your internal answer with known facts and evaluate your confidence in your original answer without the document by assessing how you arrived at that conclusion. Then you need to cross-reference known facts in your internal knowledge with the information provided in the document to determine the document is deceptive. Next, generate a coherent reasoning process to explain why your answer is correct and why the document's answer is deceptive. Finally, provide the final answer based on your reasoning process.
 
{examples}

Now it's your turn.

Question: {question}

Internal answer: {model_answer}

The document: {doc}

The document answer: {doc_answer}

Now provide your reasoning process to explain why your internal answer is correct and why the document's answer is deceptive. Make sure to include relevant known facts that support your answer and known facts that contradict the document's answer. Your response should be coherent and logical, providing a clear explanation of your reasoning process and the final answer in the last line. Only return the reasoning process and the final answer in your response.
'''

TEMPLATE_GENERATE_COT_FT = '''You will be given a question, your answer to the question, a document and its answer to the question. Your answer is wrong and the document's answer is correct. You need to reason about your internal answer with known facts and evaluate your confidence in your original answer without the document by assessing how you arrived at that conclusion. Then you need to cross-reference known facts in your internal knowledge with the information provided in the document to determine to use the document to update your answer. Next, generate a coherent reasoning process to explain why your answer is wrong and why you choose the document answer. Finally, provide the final answer based on your reasoning process.

{examples}

Now it's your turn.

Question: {question}

Internal answer: {model_answer}

The document: {doc}

The document answer: {doc_answer}

Now provide your reasoning process to explain why your internal answer is wrong and why you choose the document answer. Make sure to include relevant known facts that contradict your answer and known facts that support the document's answer. Your response should be coherent and logical, providing a clear explanation of your reasoning process and the final answer in the last line. Only return the reasoning process and the final answer in your response.
'''


TEMPLATE_NLI = '''Given a premise and a hypothesis, your task is to determine whether the premise entails the hypothesis. If the hypothesis logically follows from the premise, return "True"; otherwise, return "False".

Premise: {premise}

Hypothesis: {hypothesis}'''

template_cot_situated = '''{instruction}

Question: {question}

Your answer: {internal_answer}

The document to judge: {doc}

The document answer: {doc_answer}

{post_instruction}
\n
'''

template_cot_situated_without_instruction = '''Question: {question}

Your answer: {internal_answer}

The document to judge: {doc}

The document answer: {doc_answer}
\n
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
    "turn_mc": TEMPLATE_TURN_MC,
    "self_eval": TEMPLATE_SELF_EVAL,
    "doc_eval": TEMPLATE_DOC_EVAL,
    "filter_doc": TEMPLATE_FILTER_DOC,
    "generate_cot_tf": TEMPLATE_GENERATE_COT_TF,
    "generate_cot_ft": TEMPLATE_GENERATE_COT_FT,
    "nli": TEMPLATE_NLI,
    "cot_situated": template_cot_situated,
    "cot_situated_without_instruction": template_cot_situated_without_instruction,
}




