import re
def extract_choice_letter(answer):
    # This pattern looks for any capital letter followed by a period or a parenthesis
    match = re.search(r'\b([A-F])[\.\)]|\([A-Z]\)', answer)
    if match:
        return match.group(1)
    else:
        print("No match found for the answer:", answer)
        return None


def turn_candidates_into_multiple_choice(candidate_answers):
    # This function takes a list of candidate answers and turns them into a multiple choice question
    # by adding a letter to each answer
    multiple_choices = []
    for i, answer in enumerate(candidate_answers):
        letter = chr(65 + i)
        multiple_choices.append(f"{letter}) {answer}")
    return multiple_choices

