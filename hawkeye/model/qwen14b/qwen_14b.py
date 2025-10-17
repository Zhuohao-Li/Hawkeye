from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
from tqdm import tqdm

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", device_map="auto")  # Automatically map to available GPUs

# Input and output file paths
input_file = "./dataset/test_1.jsonl"  # Path to the dataset
output_file = "./evaluation_results.jsonl"  # Path to save the results

def evaluate_model(input_file, output_file):
    """
    Evaluate the model on a dataset and save only the model outputs.

    :param input_file: Path to the input JSONL file containing questions.
    :param output_file: Path to the output JSONL file to save results.
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        questions = [json.loads(line) for line in file]

    # Open the output file for appending
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for idx, item in enumerate(tqdm(questions, desc="Processing questions")):
            question = item.get("question", "")
            if not question:
                continue

            # Add a prompt to clarify the model's role
            prompt = ("Please provide the correct answer to the following question:\n\n" + question)

            # Tokenize the input question
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            # Generate the model's output
            outputs = model.generate(**inputs, max_length=1024, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Write only the model output to the output file
            out_file.write(json.dumps({"model_output": generated_text}, ensure_ascii=False) + "\n")

            # Clear the kv_cache to avoid memory issues
            if hasattr(model, 'clear_kv_cache'):
                model.clear_kv_cache()

    print(f"Evaluation results saved to {output_file}")

# Run the evaluation
evaluate_model(input_file, output_file)
