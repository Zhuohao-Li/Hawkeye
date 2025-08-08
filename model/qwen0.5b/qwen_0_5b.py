from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", device_map="auto")  # Automatically map to available GPUs

# 设置 pad_token_id 以避免警告
model.config.pad_token_id = tokenizer.eos_token_id

# Input and output file paths
input_file = "../dataset/test_1.jsonl"
output_file = "./baseline_evaluation_results.jsonl"

def evaluate_model(input_file, output_file):
    """
    Evaluate the model on a dataset and append results to the output file after each output.
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        questions = [json.loads(line) for line in file]

    # Open the output file in append mode
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for idx, item in enumerate(tqdm(questions, desc="Processing questions")):
            question = item.get("question", "")
            if not question:
                continue

            # Construct the prompt
            system_prompt = (
                "<|system|>You are a helpful assistant skilled in solving problems step-by-step "
                "using a chain of thought approach. Please ensure your reasoning is clear and "
                "your final answer is correct.<|user|>"
            )

            # Construct the full prompt
            prompt = (
                f"{system_prompt}Question: {question}\n\n"
                "Please think step by step and provide the correct answer.\n<|assistant|>"
            )

            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            # Generate output
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # 仅限制生成文本的长度
                num_return_sequences=1,
                do_sample=False
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt if it exists in the output
            if generated_text.startswith(prompt):
                answer_text = generated_text[len(prompt):].strip()
            else:
                answer_text = generated_text.strip()

            # Append the result to the file
            out_file.write(
                json.dumps({"model_output": answer_text}, ensure_ascii=False) + "\n"
            )

            # Clear the kv_cache if available
            if hasattr(model, 'clear_kv_cache'):
                model.clear_kv_cache()

    print(f"Evaluation results saved to {output_file}")

# Run the evaluation
evaluate_model(input_file, output_file)
