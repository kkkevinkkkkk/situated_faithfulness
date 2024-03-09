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
TEMPLATES = {
    "llama2_chat": TEMPLATE_LLAMA2_CHAT,
    "synthesize_deceptive_document": TEMPLATE_SYNTHESIZE_DECEPTIVE_DOCUMENT,
    "qa_to_statement": TEMPLATE_QA_TO_STATEMENT,
    "generate_support": TEMPLATE_GENERATE_SUPPORT,
    "generate_multiple_choice": TEMPLATE_GENERATE_MULTIPLE_CHOICE
}




