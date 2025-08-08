from datasets import load_dataset
import os
from openai import OpenAI
import re
import yaml
import json
from tqdm import tqdm
from pprint import pprint

client = OpenAI(base_url=os.environ["base_url"], api_key=os.environ["api_key"])

with open("gen_cot.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    pprint(config)
    
gen_cot_template = """
Based on the provided question: {question}, generate a detailed and coherent chain of thought to guide the process of solving this question efficiently and effectively. 

Requirements:
1. The chain of thought must focus solely on the reasoning process and step-by-step approach, avoiding any immediate answers.
2. Express the chain of thought as a numbered list using the format:
   1. ...
   2. ...
   3. ...
   
Return the output in the following JSON format:
```json
{{
    "chain_of_thought": "..."
}}
"""



def map_gsm8k_function(example):
    example["answer"] = example["answer"].split("####")[1].strip()
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
    assert float(
        example["answer"]
    ), f"answer is not a valid number: {example['answer']}"

    return example


def create_cot_prompt(question: str):
    return gen_cot_template.format(question=question)


def chat_gpt(p, model="gpt-4o"):
    res = client.chat.completions.create(
        messages=[{"role": "user", "content": p}], model=model
    )
    return res.choices[0].message.content

def custom_load_dataset(data_name: str):
    """
    :return: (dataset, question field name)
    """
    if data_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", name="main", split="test")
        ds = ds.map(map_gsm8k_function, num_proc=config["num_proc"], desc=f"Preprocessing data: {data_name} with {config['num_proc']} num_proc")
        return ds, "question"
    elif data_name == "MATH":
        ds = load_dataset("lighteval/MATH", name="all", split="test"), 
        return ds, "problem"
    else:
        raise NotImplementedError(f"{data_name} has not been implemented yet")

def extract_bracket_content(text):
    # Extract content between the first '{' and the last '}'
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None


def main():
    ds, question_field = custom_load_dataset(config["data_name"])
    if config["test"]:
        # select part of example for test
        ds = ds.shuffle(seed=config["seed"]).select(range(5))
    
    output_path = f"{config["data_name"]}_cot.jsonl"
    if os.path.exists(output_path):
        os.remove(output_path)

    # TODO(High Priority): multi-process to accelerate the for-loop logic
    with open(output_path, mode="a", encoding="utf-8") as f:
        for example in tqdm(ds, desc="Generating CoT..."):
            prompt = create_cot_prompt(example[question_field])
            for attempt in range(config["max_try_times"]):
                try:
                    cot_res = chat_gpt(prompt)
                    
                    # NOTE: Structured output was intentionally avoided as enforcing a rigid structure might degrade the model's performance.
                    cot_content = extract_bracket_content(cot_res)
                    cot = json.loads(cot_content)["chain_of_thought"]
                    break  # Exit the loop if successful
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    # NOTE: GPT-4o is enough strong to generate suitable format, this case never runs
                    # Handle specific errors like invalid JSON or missing keys
                    cot = "None"
                    if attempt == config["max_try_times"] - 1:
                        # Log the failure or raise an exception if all attempts fail
                        print(f"Failed after {config['max_try_times']} attempts: {e}")
            if isinstance(cot, list):
                cot = "\n".join(cot)
            example["cot"] = cot
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
if __name__ == "__main__":
    main()
