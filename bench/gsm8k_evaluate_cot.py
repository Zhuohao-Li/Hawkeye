import ast
from datasets import load_dataset
from tqdm import tqdm
from sglang import assistant, gen, set_default_backend, Runtime, user, function
from argparse import ArgumentParser
import numpy as np
import os
import re
import json
import random
import torch

prompt_template = """
Question: {question}
"""

prompt_template_cot = """
Below is a question followed by a chain of thought. Please answer the question based on the reasoning provided in the chain of thought.

Question: {question}

Chain of Thought: {cot}
"""

INVALID = -9999999

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

@function
def qa(s, p):
    s += user(p)
    s += assistant(gen(name="output", 
                            max_tokens=args.max_new_tokens, 
                            temperature=0, stop=["Question", "Assistant:", "<|separator|>"]))
    s += user("Please provide the answer directly.")
    s += assistant("Answer: " + gen(name="answer", max_tokens=16, temperature=0, stop=["Question", "Assistant:", "<|separator|>"]))
    if args.first:
        with open(os.path.join(output_path, f"example{suffix}.txt"), mode="w", encoding="utf-8") as fp:
            fp.write(s.text())
        args.first = False
            

def create_prompt(example, is_cot=False):
    if is_cot:
        return prompt_template_cot.format(question=example["question"], cot=example["cot"])
    return prompt_template.format(question=example["question"]) 

def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID    


def parse_args():
    parser = ArgumentParser(description="Parse arguments for testing")

    parser.add_argument("--seed", type=int, default=42)
    
    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained model or model identifier from HuggingFace Model Hub.")
    
    # Chain of Thought (CoT) configuration
    parser.add_argument("--cot", action="store_true", help="Enable Chain of Thought reasoning for the model.")
    
    # Inference configurations
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    
    return parser.parse_args()

def main():
    seed_everything(args.seed)
    ds = load_dataset("json", data_files={"test": "gsm8k_cot.jsonl"}, split="test")

    runtime = Runtime(model_path=args.model_name_or_path, tp_size=1)
    set_default_backend(backend=runtime)

    num_batches = (len(ds) - 1) // args.batch_size + 1
    preds = []
    for batch in tqdm(range(num_batches)):
        start = batch * args.batch_size
        end = min((batch + 1) * args.batch_size, len(ds)) 
        states = qa.run_batch([{
            "p": create_prompt(example=ds[i], is_cot=args.cot)
        } for i in range(start,end)])
        preds.extend(get_answer_value(s["answer"]) for s in states)

    # Compute metrics
    labels = [ast.literal_eval(answer) for answer in ds["answer"]]
    results = np.array(preds) == np.array(labels) 
    invalid_ratio = float(np.mean(np.array(preds) == INVALID))  # Fraction of predictions that are marked as INVALID
    correct_count = int(np.sum(results))  # Total number of correct 
    wrong_count = int(np.sum(~results))  # Total number of incorrect predictions
    accuracy = float(np.mean(results))  # Overall accuracy

    # Display metrics for debugging or logging
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct predictions: {correct_count}")
    print(f"Wrong predictions: {wrong_count}")
    print(f"Invalid predictions ratio: {invalid_ratio:.4f}")

    # write experiment args and results
    with open(os.path.join(output_path, f"gsm8k_args{suffix}.json"), mode="w", encoding="utf-8") as fout:
        args_dict = vars(args)
        json.dump(args_dict, fout, indent=4)
    with open(os.path.join(output_path, f"gsm8k_result{suffix}.json"), mode="w", encoding="utf-8") as fout:
        json.dump({
            "invalid_ratio": invalid_ratio,
            "correct_count": correct_count,
            "wrong_count": wrong_count,
            "accuracy": accuracy
        }, fp=fout, indent=4)
    runtime.shutdown()

if __name__ == "__main__":
    args = parse_args()
    args.first = True
    # create dir containing the model name to store the results
    model_name_or_path = args.model_name_or_path.strip("/ ")
    model_name = model_name_or_path.split("/")[-1]
    output_path = f"results/{model_name}"
    suffix = "_cot" if args.cot else ""
    os.makedirs(output_path, exist_ok=True)
    main()


