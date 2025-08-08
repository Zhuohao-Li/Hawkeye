from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
from tqdm import tqdm

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", device_map="auto")  # Automatically map to available GPUs

# File paths
#input_questions_file = "./dataset/test_1.jsonl"  # Path to the original dataset with questions
input_questions_file = "./dataset/q120.jsonl" 
chain_of_thought_file = "./dataset/chain_of_thought_output.jsonl"  # Path to the chain of thought file
output_file = "./evaluation_results_with_cot.jsonl"  # Path to save the results

def load_chain_of_thought(file_path):
    """Load chain of thought data from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return {json.loads(line)["question"]: json.loads(line)["chain_of_thought"] for line in file}

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

    # Open the output file for appending
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for idx, item in enumerate(tqdm(questions, desc="Processing questions")):
            question = item.get("question", "")
            if not question or question not in chain_of_thoughts:
                continue

            chain_of_thought = chain_of_thoughts[question]

            # Construct the prompt with question and chain of thought
            prompt = (f"You are a math-solving assistant. Below is a question and a chain of thought. The chain of thought is correct, you need to answer based on the chain of thought. "
                      f"Using this reasoning chain and provide the final correct answer:\n\n"
                      f"Question: {question}\n\n"
                      f"Chain of Thought: {chain_of_thought}\n\n"
                      f"Answer:")

            # Tokenize the input prompt
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
evaluate_model_with_cot(input_questions_file, chain_of_thought_file, output_file)
