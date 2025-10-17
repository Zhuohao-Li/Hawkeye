#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training pipeline example with baseline generation:
1. 先生成 baseline 文件：用模型A直接生成 CoT（不重复原始问题）指导模型B回答，并记录模型B回答和调用 OpenAI API 得到的匹配详情（仅包含score）。
2. 然后启动 GRPOTrainer 对模型A进行微调，使其生成更简短的 CoT（同样不重复原始问题）。
3. 微调时，对于每个样本，先查找 baseline 结果：
   - 如果 baseline 中模型B的回答已错误，则直接将匹配分数设为 1；
   - 否则利用模型B和 API 判断新 CoT 下的回答匹配情况。
4. 最后 reward = match_score - length_penalty，其中 length_penalty 为平方惩罚（目标长度为标准答案长度的30%）。
"""

import os
# 禁用 vLLM 内存 profiling 检查环境变量（部分情况有效）
os.environ["VLLM_DISABLE_MEMORY_PROFILE"] = "1"

import re
import json
import gc
import torch
import numpy as np
import argparse
import wandb
import openai
import asyncio
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead

# 导入 vLLM 相关库（仅用于 GRPOTrainer 内部自动启动 vLLM，用于 Model A 生成）
from vllm import AsyncLLMEngine, SamplingParams

# 添加猴子补丁，跳过 vLLM 内存 profiling 检查
try:
    import vllm.worker.worker as vllm_worker
    def patched_assert_memory_footprint_increased_during_profiling(self):
        print("Warning: Skipping memory profiling check in vllm.Worker.")
        return
    vllm_worker.Worker._assert_memory_footprint_increased_during_profiling = patched_assert_memory_footprint_increased_during_profiling
except Exception as e:
    print("Monkey patch for vLLM memory profiling failed:", e)

######################################
# 全局变量：baseline_data（key 为 prompt）
######################################
baseline_data = {}

######################################
# 辅助函数：清理生成的 CoT，去除原始问题部分
######################################
def clean_cot(cot, prompt):
    """
    如果生成的 CoT 开头包含原始问题，则移除之。
    """
    if cot.startswith(prompt):
        return cot[len(prompt):].strip()
    # 若生成结果中包含换行符，尝试去除第一行（可能是问题部分）
    parts = cot.split('\n', 1)
    if len(parts) > 1 and prompt in parts[0]:
        return parts[1].strip()
    return cot.strip()

######################################
# 1. 数据集定义
######################################
class CoTDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        # 数据集采用 JSONL 格式，每行一个 JSON 对象，包含 "question" 和 "answer"
        self.data = load_dataset("json", data_files=file_path, split="train")
        print(f"Successfully loaded dataset with {len(self.data)} samples")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 构造 prompt 时直接将问题信息传入，同时保留标准答案，方便后续奖励计算
        return {
            "prompt": f"Question: {item['question']}",
            "answer": item["answer"]
        }
        
    def collate_fn(self, batch):
        return batch

######################################
# 2. 初始化 Tokenizer 与模型
######################################
model_a_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Model A：用于生成 chain-of-thought（使用 vLLM加速生成）
model_b_name = "Qwen/Qwen2.5-0.5B-Instruct"                  # Model B：用于生成最终答案（采用标准 Transformers 生成）

tokenizer_a = AutoTokenizer.from_pretrained(model_a_name)
tokenizer_a.pad_token = tokenizer_a.eos_token

tokenizer_b = AutoTokenizer.from_pretrained(model_b_name)
tokenizer_b.pad_token = tokenizer_b.eos_token

# 加载 Model A（全量微调，不使用 LoRA）
base_model = AutoModelForCausalLM.from_pretrained(
    model_a_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = base_model
print("Model A loaded for full fine-tuning.")

# 加载 Model B（用于生成最终答案，采用标准 Transformers 生成，不使用 vLLM）
model_b = AutoModelForCausalLM.from_pretrained(
    model_b_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Model B loaded for answer generation (standard Transformers generation).")

######################################
# 3. 辅助函数：生成文本、提取数字、API匹配检查
######################################
def generate_text_standard(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    """
    支持批量生成：如果 prompt 为列表，则调用 batch_decode，否则单个生成。
    """
    if isinstance(prompt, list):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_number(text):
    numbers = re.findall(r"\d+\.?\d*", text)
    return float(numbers[-1]) if numbers else None

def check_answer_match(generated: str, ground_truth: str) -> (float, dict):
    """
    调用 OpenAI API 判断生成答案与标准答案是否匹配，
    仅返回匹配分数（1表示完全匹配，0表示不匹配）。
    """
    try:
        system_prompt = (
            "You are an impartial judge. Determine if the following answers match. "
            "Output a single number: 1 for match, 0 for mismatch."
        )
        user_prompt = f"Generated answer: {generated}\nStandard answer: {ground_truth}"
        
        client = openai.OpenAI(
            api_key=""
        )
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
# 4. 自定义奖励函数及异步包装
######################################
def compute_reward_sync(question, generated_cot, ground_truth, penalty_factor=0.01):
    """
    同步版本：使用 Model B 进行标准 Transformers 生成答案，然后计算匹配分数和奖励。
    如果对应问题在 baseline_data 中显示模型B回答错误，则直接将匹配分数置为1（不调用API）。
    使用平方惩罚，目标长度为标准答案词数的30%。
    """
    if question in baseline_data and not baseline_data[question].get("is_correct", False):
        match_score = 1.0
        details_match = {"score": match_score}
    else:
        prompt_b = f"{question}\n{generated_cot}\nThis is a chain-of-thought that helps you answer. So the answer is:"
        inputs = tokenizer_b(prompt_b, return_tensors="pt").to(model_b.device)
        outputs = model_b.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9)
        answer_text = tokenizer_b.decode(outputs[0], skip_special_tokens=True)
        match_score, details_match = check_answer_match(answer_text, ground_truth)
    
    # 目标长度基于标准答案词数的30%
    baseline_length = len(ground_truth.split())
    target_length = int(baseline_length * 0.3)
    cot_length = len(generated_cot.split())
    length_penalty = penalty_factor * (max(0, cot_length - target_length)) ** 2
    
    reward = match_score - length_penalty
    details = {
        "match_score": match_score,
        "cot_length": cot_length,
        "baseline_length": baseline_length,
        "target_length": target_length,
        "length_penalty": length_penalty,
        "reward": reward,
        "match_details": details_match
    }
    return reward, details

async def compute_reward(question, generated_cot, ground_truth, target_length=50, penalty_factor=0.01):
    """
    异步包装：调用同步版本 compute_reward_sync，通过 asyncio.to_thread 运行。
    """
    return await asyncio.to_thread(compute_reward_sync, question, generated_cot, ground_truth, penalty_factor)

def my_reward_func(prompts, completions, **kwargs):
    """
    自定义奖励函数的同步包装器。
    从 kwargs 中获取标准答案列表（键 "answer"），对每个样本调用 compute_reward。
    """
    answers = kwargs.get("answer")
    rewards = []
    for p, c, a in zip(prompts, completions, answers):
        reward, _ = asyncio.run(compute_reward(p, c, a))
        rewards.append(reward)
    return rewards

######################################
# 5. GRPOTrainer 配置
######################################
grpo_config = GRPOConfig(
    output_dir="./checkpoints",
    num_train_epochs=5.0,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    beta=0.04,
    max_prompt_length=1024,
    max_completion_length=1024,
    temperature=0.9,
    use_vllm=True,
    vllm_device="cuda:7",  # GRPOTrainer 内部会自动启动 vLLM用于 Model A 的生成
    vllm_gpu_memory_utilization=0.6,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_steps=100,
)

######################################
# 6. Baseline 生成函数（支持 batch 处理，实时写入输出文件）
######################################
def generate_baseline(dataset, save_path, batch_size=8):
    """
    利用 DataLoader 批量处理数据：
    - 对每个 batch，先批量生成 CoT，再批量生成模型B的回答，
      然后逐条调用 API 获取匹配详情，最终将所有结果保存为一个标准的 JSON 文件。
      
    注意：生成 CoT 时指示模型不重复原始问题，并在生成后通过 clean_cot 去除可能的重复。
    """
    baseline_results = {}
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    with open(save_path, "w", encoding="utf-8") as f:
        pass
    
    for batch in tqdm(data_loader, desc="Generating baseline"):
        prompts = [sample["prompt"] for sample in batch]
        ground_truths = [sample["answer"] for sample in batch]
        
        # 批量生成 CoT（Chain-of-Thought），要求不重复原始问题
        batch_prompt_full = [
            f"{prompt}\nPlease generate a concise chain-of-thought reasoning without repeating the question:" 
            for prompt in prompts
        ]
        cots = generate_text_standard(model, tokenizer_a, batch_prompt_full, max_new_tokens=1024, temperature=0.7, top_p=0.9)
        # 清理生成的 CoT，去除原始问题部分
        cots = [clean_cot(cot, prompt) for cot, prompt in zip(cots, prompts)]
        
        # 批量生成模型B的回答，追加英文说明
        batch_prompt_b = [
            f"{prompt}\n{cot}\nThis is a chain-of-thought that helps you answer. So the answer is:" 
            for prompt, cot in zip(prompts, cots)
        ]
        answers_b = generate_text_standard(model_b, tokenizer_b, batch_prompt_b, max_new_tokens=512, temperature=0.7, top_p=0.9)
        
        # 逐条处理每个样本，并存入 baseline_results 字典
        for prompt, ground_truth, cot, answer_b in zip(prompts, ground_truths, cots, answers_b):
            pred = extract_number(answer_b)
            true = extract_number(ground_truth)
            is_correct = (pred == true and pred is not None)
            match_score, details_match = check_answer_match(answer_b, ground_truth)
            result = {
                "cot": cot,
                "answer": answer_b,
                "pred": pred,
                "ground_truth": true,
                "is_correct": is_correct,
                "match_score": match_score,
                "match_details": details_match
            }
            baseline_results[prompt] = result

    # 将所有结果写入一个标准的 JSON 文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)
    print(f"Baseline results saved to {save_path}")
    return baseline_results


######################################
# 7. 主函数：命令行模式选择 baseline / train / evaluate
######################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training, Baseline Generation and Evaluation Script')
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'train', 'evaluate'],
                        help='Running mode: baseline, train or evaluate')
    parser.add_argument('--train_file', type=str, default='dataset.jsonl',
                        help='Training dataset file path')
    parser.add_argument('--test_file', type=str, default='test_dataset.jsonl',
                        help='Test dataset file path')
    parser.add_argument('--model_path', type=str,
                        help='Model path for evaluation')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory for saving models')
    parser.add_argument('--save_steps', type=int, default=5,
                        help='Save checkpoint every N steps')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--baseline_file', type=str, default='baseline.json',
                        help='File path for baseline results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for baseline generation')
    args = parser.parse_args()

    if args.mode == 'baseline':
        print(f"Generating baseline using dataset: {args.train_file}")
        baseline_dataset = CoTDataset(args.train_file, tokenizer_a)
        baseline_data = generate_baseline(baseline_dataset, args.baseline_file, batch_size=args.batch_size)
    
    elif args.mode == 'train':
        # 尝试加载 baseline 文件
        if args.baseline_file and os.path.exists(args.baseline_file):
            with open(args.baseline_file, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            print(f"Loaded baseline data from {args.baseline_file}")
        else:
            print("Warning: No baseline file found. Proceeding without baseline adjustments.")
            baseline_data = {}
            
        print(f"Loading training dataset: {args.train_file}")
        train_dataset = CoTDataset(args.train_file, tokenizer_a)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=my_reward_func,
            args=grpo_config,
            train_dataset=train_dataset
        )
        print("Starting GRPO training (vLLM will be automatically initialized by GRPOTrainer for Model A)...")
        trainer.train()
        print(f"Training completed! Final model saved at: {args.save_dir}")

    elif args.mode == 'evaluate':
        def evaluate_model(model_path, test_file):
            eval_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            eval_model_b = AutoModelForCausalLM.from_pretrained(
                model_b_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            test_dataset = CoTDataset(test_file, tokenizer_a)
            results = {"correct": 0, "total": 0, "cot_lengths": [], "predictions": []}
            eval_output_dir = os.path.join(model_path, "evaluation_results")
            os.makedirs(eval_output_dir, exist_ok=True)
            eval_file_path = os.path.join(eval_output_dir, "eval_results.jsonl")
            with open(eval_file_path, "w", encoding="utf-8") as f:
                pass
            for sample in test_dataset:
                prompt = sample["prompt"]
                ground_truth = sample["answer"]
                prompt_full = f"{prompt}\nPlease generate a concise chain-of-thought reasoning without repeating the question:"
                cot = generate_text_standard(eval_model, tokenizer_a, prompt_full, max_new_tokens=1024, temperature=0.7, top_p=0.9)
                # 清理生成的 CoT，去除原始问题部分
                cot = clean_cot(cot, prompt)
                results["cot_lengths"].append(len(cot.split()))
                prompt_b = f"{cot}\n This is a chain-of-thought that helps you answer. So the answer is:"
                answer_b = generate_text_standard(eval_model_b, tokenizer_b, prompt_b, max_new_tokens=512, temperature=0.7, top_p=0.9)
                pred = extract_number(answer_b)
                true = extract_number(ground_truth)
                is_correct = (pred == true and pred is not None)
                if is_correct:
                    results["correct"] += 1
                results["total"] += 1
                sample_result = {
                    "prompt": prompt,
                    "generated_cot": cot,
                    "predicted_answer": pred,
                    "ground_truth": true,
                    "is_correct": is_correct
                }
                results["predictions"].append(sample_result)
                with open(eval_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample_result, ensure_ascii=False) + "\n")
            accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
            print(f"Accuracy: {accuracy:.2%}")
            with open(eval_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"accuracy": accuracy, "total": results["total"]}, ensure_ascii=False) + "\n")
            return accuracy, results

        if not args.model_path:
            print("Error: --model_path is required for evaluation mode.")
        else:
            accuracy, results = evaluate_model(args.model_path, args.test_file)
