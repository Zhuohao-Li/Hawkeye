#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for a model using OpenAI's API for answer matching.
Supports two evaluation modes:
- single: only one model is used for both chain-of-thought (CoT) and answer generation.
- joint: a big model is used for CoT generation and a small model (0.5B) is used for answer generation.
"""

import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

import openai

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

######################################
# Helper function: clean chain-of-thought
######################################
def clean_cot(cot, prompt):
    if cot.startswith(prompt):
        return cot[len(prompt):].strip()
    parts = cot.split('\n', 1)
    if len(parts) > 1 and prompt in parts[0]:
        return parts[1].strip()
    return cot.strip()

######################################
# Dataset definition
######################################
class CoTDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = load_dataset("json", data_files=file_path, split="train")
        print(f"Successfully loaded dataset with {len(self.data)} samples")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": f"Question: {item['question']}, please generate a chain-of-thought reasoning without repeating the question, make sure the reasoning is correct and simple:",
            "answer": item["answer"]
        }

######################################
# Helper function: generate text
######################################
def generate_text_standard(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    if isinstance(prompt, list):
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    else:
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

######################################
# OpenAI API match checking function
######################################
def check_answer_match(generated: str, ground_truth: str, llm_judge: bool, api_key: str) -> (float, dict):
    try:
        system_prompt = (
            "You are an impartial judge. Determine if the following answers match. "
            "Output a single number: 1 for match, 0 for mismatch."
        )
        user_prompt = f"Generated answer: {generated}\nStandard answer: {ground_truth}"
        # Write OpenAI API key in .env file
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
        return score, {"score": score}
    except Exception as e:
        print(f"OpenAI API call error: {str(e)}")
        return 0.0, {"score": 0.0, "error": str(e)}

######################################
# Evaluation logic
######################################
def evaluate_model(model_path, test_file, eval_mode="joint", is_local=False, api_key=""):
    print("Starting evaluation...")

    # For debugging, print whether the model_path exists
    print("os.path.exists(model_path):", os.path.exists(model_path))
    
    # Determine if model_path is local: either explicitly set or if the path exists and starts with "./" or "/"
    is_local = is_local or (os.path.exists(model_path) and model_path.startswith(("./", "/")))
    
    # Load big model
    if is_local:
        print("Loading big model (local) from:", model_path)
        tokenizer_big = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        eval_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("Loading big model (HF) from:", model_path)
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

    print("Loading test dataset:", test_file)
    test_dataset = CoTDataset(test_file, tokenizer_big)
    print(f"Test dataset contains {len(test_dataset)} samples.")

    # Save evaluation results:
    if is_local:
        eval_output_dir = os.path.join(model_path, "evaluation_results")
        os.makedirs(eval_output_dir, exist_ok=True)
        eval_file_path = os.path.join(eval_output_dir, "eval_results.jsonl")
    else:
        model_name_for_file = model_path.replace("/", "_")
        eval_file_path = os.path.join(os.getcwd(), f"{model_name_for_file}_eval_results.jsonl")
    
    # Clear result file
    with open(eval_file_path, "w", encoding="utf-8") as f:
        pass

    results = {"correct": 0, "total": 0, "cot_lengths": [], "answer_lengths": [], "predictions": []}
    for idx, sample in enumerate(tqdm(test_dataset, desc="Evaluating samples")):
        prompt = sample["prompt"]
        ground_truth = sample["answer"]
        prompt_full = f"{prompt}\nPlease generate a concise chain-of-thought reasoning without repeating the question:"
        cot = generate_text_standard(eval_model, tokenizer_big, prompt_full, max_new_tokens=1024, temperature=0.7, top_p=0.9)
        cot = clean_cot(cot, prompt)
        results["cot_lengths"].append(len(cot.split()))

        prompt_b = f"{cot}\n This is a chain-of-thought that helps you answer. So the answer is:"
        answer_b = generate_text_standard(eval_model_small, tokenizer_small, prompt_b, max_new_tokens=512, temperature=0.7, top_p=0.9)
        results["answer_lengths"].append(len(answer_b.split()))
        
        match_score, details_match = check_answer_match(answer_b, ground_truth, True, api_key)
        is_correct = (match_score == 1.0)
        if is_correct:
            results["correct"] += 1
        results["total"] += 1
        
        sample_result = {
            "prompt": prompt,
            "generated_cot": cot,
            "generated_answer": answer_b,
            "ground_truth": ground_truth,
            "match_score": match_score,
            "match_details": details_match,
            "is_correct": is_correct,
            "cot_length": len(cot.split()),
            "answer_length": len(answer_b.split())
        }
        results["predictions"].append(sample_result)
        with open(eval_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample_result, ensure_ascii=False) + "\n")
    
    accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
    avg_cot_length = sum(results["cot_lengths"]) / len(results["cot_lengths"]) if results["cot_lengths"] else 0
    avg_answer_length = sum(results["answer_lengths"]) / len(results["answer_lengths"]) if results["answer_lengths"] else 0
    print(f"Final Accuracy: {accuracy:.2%}")
    print(f"Average CoT Length: {avg_cot_length:.2f} words")
    print(f"Average Answer Length: {avg_answer_length:.2f} words")
    
    with open(eval_file_path, "a", encoding="utf-8") as f:
        final_stats = {
            "accuracy": accuracy,
            "total": results["total"],
            "avg_cot_length": avg_cot_length,
            "avg_answer_length": avg_answer_length
        }
        f.write(json.dumps(final_stats, ensure_ascii=False) + "\n")
    return accuracy, results

######################################
# Main: parse arguments and run evaluation
######################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script for a model using OpenAI API for answer matching.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint or Hugging Face model ID for the big model.')
    parser.add_argument('--test_file', type=str, default='test_dataset.jsonl',
                        help='Test dataset file path.')
    parser.add_argument('--eval_mode', type=str, choices=["single", "joint"], default="joint",
                        help='Evaluation mode: "single" to evaluate one model; "joint" to use big model + small model for joint inference.')
    parser.add_argument('--is_local', action='store_true',
                        help='If set, load model using local path (e.g., fine-tuned model).')
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    acc, res = evaluate_model(args.model_path, args.test_file, eval_mode=args.eval_mode, is_local=args.is_local, api_key=api_key)
    print("Evaluation completed. Accuracy: {:.2%}".format(acc))

# 使用 HuggingFace 模型进行评估
#python evaluation.py --model_path Qwen/Qwen1.5-7B --test_file test_data.jsonl --eval_mode joint

# 使用本地模型进行评估
#python evaluation.py --model_path ./checkpoints/checkpoint-50 --test_file ./dataset/test_dataset.jsonl --eval_mode single --is_local

#accelerate launch --num_processes=8 evaluation.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --test_file ./dataset/test_dataset.jsonl --eval_mode single
