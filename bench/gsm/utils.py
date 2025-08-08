import random
import re
import ast
import torch
import numpy as np
import os


INVALID = -99999

cot_prompt = """Question: {question}. Please generate a concise chain-of-thought reasoning without repeating the question.
"""

doc_to_text = """Question: {question}
Cot: {cot}
This is a chain-of-thought that helps you answer.
So the answer is:
"""


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def map_function(example):
    example["answer"] = example["answer"].split("####")[1].strip()
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    example["answer"] = re.sub(
        pattern=r"(\d),(\d)", repl=r"\1\2", string=example["answer"]
    )
    assert float(
        example["answer"]
    ), f"answer is not a valid number: {example['answer']}"
    return example


def extract_answer(answer_str):
    """tranfer an output string to an exact number"""

    answer_str = re.sub(r"(\d),(\d)", r"\1\2", answer_str)

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_str)

    return numbers[-1] if numbers else INVALID
