from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto")  # Automatically map to GPUs if possible

# File paths
input_questions_file = "../dataset/test_1.jsonl"       # Path to the original dataset with questions
chain_of_thought_file = "../dataset/chain_of_thought_output.jsonl"  # Path to the chain of thought file
output_file = "./evaluation_results_with_cot.jsonl"     # Path to save the results

def load_chain_of_thought(file_path):
    """Load chain of thought data from a JSONL file."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            q = record["question"]
            cot = record["chain_of_thought"]
            data[q] = cot
    return data

def evaluate_model_with_cot(input_questions_file, chain_of_thought_file, output_file):
    """
    Evaluate the model using questions and corresponding chains of thought.
    :param input_questions_file: Path to the input JSONL file containing questions.
    :param chain_of_thought_file: Path to the JSONL file containing chains of thought.
    :param output_file: Path to the output JSONL file to save results.
    """
    # Load questions and chain of thought data
    with open(input_questions_file, 'r', encoding='utf-8') as file:
        questions = [json.loads(line) for line in file]
    chain_of_thoughts = load_chain_of_thought(chain_of_thought_file)

    # Open the output file for appending (or 'w' if you want to overwrite)
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for idx, item in enumerate(tqdm(questions, desc="Processing questions")):
            question = item.get("question", "")
            if not question or question not in chain_of_thoughts:
                # 如果题目为空，或者题目在 COT 文件里不存在，则跳过
                continue

            chain_of_thought = chain_of_thoughts[question]

            # 构造 Prompt，在最后放一个 "Answer:"，让模型知道要在这儿输出
            prompt = (
                "Below is a question and a correct chain of thought.\n"
                "Please think step by step based on the chain of thought, provide the final correct answer.\n\n"
                f"Question: {question}\n\n"
                f"Chain of Thought:\n{chain_of_thought}\n\n"
                "Answer:"
            )

            # Tokenize the input prompt
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            # Generate the model's output
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,  # 仅限制生成文本的长度
                num_return_sequences=1,
                do_sample=False
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 方法1：基于 "Answer:" 切割
            parts = generated_text.split("Answer:")
            if len(parts) > 1:
                # 取最后一个 split 结果
                answer_text = parts[-1].strip()
            else:
                # 万一模型连 "Answer:" 都没有生成
                answer_text = generated_text.strip()

            # 如果还是空的，就先不截了，看能不能保留更多内容
            if not answer_text:
                answer_text = generated_text.strip()

            # 写入输出文件
            out_file.write(json.dumps({
                "index": idx + 1,
                "model_output": answer_text
            }, ensure_ascii=False) + "\n")

            # Clear the kv_cache to avoid memory issues (if Qwen supports it)
            if hasattr(model, 'clear_kv_cache'):
                model.clear_kv_cache()

    print(f"Evaluation results saved to {output_file}")


# 如果你需要直接跑，可以在主流程里调用
if __name__ == "__main__":
    evaluate_model_with_cot(input_questions_file, chain_of_thought_file, output_file)
