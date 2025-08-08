#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized evaluation script with batch processing and separated system/user prompts.
"""

import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Optional, List
import openai
import concurrent.futures

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

######################################
# Helper function: clean chain-of-thought
######################################
def clean_cot(cot: str, prompt: str) -> str:
    if cot.startswith(prompt):
        return cot[len(prompt):].strip()
    parts = cot.split('\n', 1)
    if len(parts) > 1 and prompt in parts[0]:
        return parts[1].strip()
    return cot.strip()

######################################
# Dataset definition with separated prompts
######################################
class CoTDataset(Dataset):
    def __init__(self, ds_name: str):
        self.data = load_dataset(ds_name, split="test")
        print(f"Loaded dataset with {len(self.data)} samples")
        
        # Define system prompts
        self.cot_system_prompt = """You are a mathematical reasoning expert. Break down the problem into clear, step-by-step reasoning by explicitly outlining each intermediate calculation or logical deduction. Do not restate the problem; focus solely on constructing a coherent chain of thought that leads directly to the solution.
        
        Problem:
        {}
        
        <think>
        """.strip()
        
        self.answer_system_prompt = """Based on the provided reasoning, solve the following math problem efficiently and clearly. Incorporate the given reasoning into your step-by-step process before arriving at your answer. The final line of your response must be exactly: 'Therefore, the final answer is: \(\boxed{{ANSWER}}\). I hope it is correct' (without quotes), where ANSWER is just the final number or expression that solves the problem.

        Problem:
        {}
        
        Reasoning:
        <think>
        {}
        
        Final Answer:
        """.strip()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "problem": item["problem"],
            "answer": item["answer"],
            "cot_messages": self.cot_system_prompt.format(item["problem"]),
            "answer_messages": self.answer_system_prompt
        }

def collate_fn(batch):
    return {
        "problems": [item["problem"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "cot_messages": [item["cot_messages"] for item in batch],
        "answer_messages": [item["answer_messages"] for item in batch]
    }

######################################
# Batch generation functions
######################################

# Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent endless repetitions or incoherent outputs.
def batch_generate(
    model,
    tokenizer,
    messages_batch: List[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9):
    device = model.device
    
    inputs = tokenizer(
        messages_batch, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
    ).to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )
    
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    token_lengths = []
    
    for i, output in enumerate(results):
        results[i] = output[len(messages_batch[i]):]
        
        tokens = tokenizer.encode(results[i], add_special_tokens=False)
        token_lengths.append(len(tokens))
        
    return results, token_lengths
    
    

######################################
# Answer verification with batch support
######################################
def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

def is_equiv(s1: str, s2: str) -> bool:
    s1_clean = s1.replace(" ", "")
    s2_clean = s2.replace(" ", "")
    return s1_clean == s2_clean
    
def process_answer(generated, ground_truth):
    ans = last_boxed_only_string(generated)
    if ans is not None:
        ans = remove_boxed(ans)
        if is_equiv(ans, ground_truth):
            return 1.0  

    try:
        system_prompt = (
            "You are an impartial judge. Determine if the following answers match. "
            "Output a single number: 1 for match, 0 for mismatch."
        )
        user_prompt = f"Generated answer: {generated}\nStandard answer: {ground_truth}"
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        response_text = response.choices[0].message.content
        score_match = re.search(r'([0-9]*\.?[0-9]+)', response_text)
        score = float(score_match.group(1)) if score_match else 0.0
        return score
    except Exception as e:
        print(f"OpenAI API call error: {str(e)}")
        return 0.0

def check_answer_match(generated_strings, ground_truth_strings):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_answer, generated_strings, ground_truth_strings)
    scores = list(results)
    return scores


######################################
# Optimized evaluation logic
######################################
def evaluate_model(
    model_path: str,
    ds_name: str,
    eval_mode: str = "joint",
    is_local: bool = False,
    batch_size: int = 8
):
    print("Initializing evaluation...")
    
    # Model loading
    if is_local or os.path.exists(model_path):
        print(f"Loading local model from {model_path}")
        tokenizer_big = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        eval_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print(f"Loading HF model: {model_path}")
        tokenizer_big = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        eval_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
    print("Big model loaded.")
    
    # Load small model for joint inference or reuse big model in single mode
    if eval_mode == "joint":
        eval_model_small = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        tokenizer_small = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=False)
        print("Small model loaded for joint inference.")
    else:
        eval_model_small = eval_model
        tokenizer_small = tokenizer_big
        print("Using single model for both chain-of-thought and answer generation.")

    # Load dataset
    print("Loading test dataset:", ds_name)
    dataset = CoTDataset(ds_name)
    print(f"Test dataset contains {len(dataset)} samples.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    # Prepare output
    output_dir = "./eval_results"
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{model_path.replace('/', '_')}_results.jsonl")

    # Evaluation metrics
    total_correct = 0
    total_samples = 0
    cot_lengths = []
    answer_lengths = []
    
    tot = 0

    for batch in tqdm(dataloader, desc="Processing batches"):
        # Generate CoT
        cot_outputs, cot_length = batch_generate(
            eval_model,
            tokenizer_big,
            batch["cot_messages"],
            max_new_tokens=1024
        )
    
        # Generate answers
        answer_messages = [
            message.format(problem, cot)
            for message, problem, cot in zip(batch["answer_messages"], batch["problems"], cot_outputs)
        ]
        
        final_answers, answer_length = batch_generate(
            eval_model_small,
            tokenizer_small,
            answer_messages,
            max_new_tokens=512
        )
        
        # Verify answers
        scores = check_answer_match(final_answers, batch["answers"])
        
        # Update metrics
        batch_correct = sum(scores)
        total_correct += batch_correct
        total_samples += len(scores)
        cot_lengths += cot_length
        answer_lengths += answer_length
        
        print(f"batch_correct: {batch_correct}, total_correct: {total_correct}, total_samples: {total_samples}")
        
        # Save results
        with open(result_path, "a", encoding="utf-8") as f:
            for i in range(len(batch["answers"])):
                record = {
                    "problem": batch["problems"][i],
                    "generated_cot": cot_outputs[i],
                    "final_answer": final_answers[i],
                    "answer_messages": answer_messages[i],
                    "ground_truth": batch["answers"][i],
                    "score": scores[i],
                    "cot_length": cot_length[i],
                    "answer_length": answer_length[i],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                

    # Final metrics
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_cot_length = sum(cot_lengths)/len(cot_lengths) if cot_lengths else 0
    avg_answer_length = sum(answer_lengths)/len(answer_lengths) if answer_lengths else 0

    print(f"\nFinal Accuracy: {accuracy:.2%}")
    print(f"Average CoT Length: {avg_cot_length:.1f} tokens")
    print(f"Average Answer Length: {avg_answer_length:.1f} tokens")

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Batch Evaluation Script")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--ds_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--eval_mode", choices=["single", "joint"], default="single")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--is_local", action="store_true")
    
    args = parser.parse_args()
    
    accuracy = evaluate_model(
        args.model_path,
        args.ds_name,
        args.eval_mode,
        args.is_local,
        args.batch_size
    )
    print(f"Evaluation completed. Accuracy: {accuracy:.2%}")