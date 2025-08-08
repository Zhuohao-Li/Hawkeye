import random
import re
import os
import numpy as np
import torch
import datasets


cot_prompt = """Question: {question}. Please generate a concise chain-of-thought reasoning without repeating the question.
"""

base_prompt_template = """What is the correct answer to this question:{Question}
Choices:
(A) {choice1}
(B) {choice2}
(C) {choice3}
(D) {choice4}"""

doc_to_text = """What is the correct answer to this question:{Question}
Choices:
(A) {choice1}
(B) {choice2}
(C) {choice3}
(D) {choice4}
Cot: {cot}
This is a chain-of-thought that helps you answer.
So the answer is:
"""
doc_to_choice = ["(A)", "(B)", "(C)", "(D)"]


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)


def extract_answer(txt):
    match = re.search(r"\((A|B|C|D)\)", txt)
    if match:
        return match.group()
    return "(ERROR)"
