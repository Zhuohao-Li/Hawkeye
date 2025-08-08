from datasets import load_dataset
from utils import (
    doc_to_text,
    cot_prompt,
    extract_answer,
    map_function,
    INVALID,
    seed_everything,
)
from sglang import assistant, set_default_backend, Runtime, function, user, gen
import yaml
import numpy as np
from tqdm import tqdm
import json
import os
import openai
import evaluate
import re


em = evaluate.load("exact_match")
client = openai.OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"])
save_dir = ""

with open("run.yaml", "r") as f:
    config = yaml.safe_load(f)


def check_answer_match(generated: str, ground_truth: str):
    try:
        system_prompt = (
            "You are an impartial judge. Determine if the following answers match. "
            "Output a single number: 1 for match, 0 for mismatch."
        )
        user_prompt = f"Generated answer: {generated}\nStandard answer: {ground_truth}"
        # Write OpenAI API key in .env file
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        response_text = response.choices[0].message.content
        score_match = re.search(r"([0-9]*\.?[0-9]+)", response_text)
        score = float(score_match.group(1)) if score_match else 0.0
        return score
    except Exception as e:
        print(f"OpenAI API call error: {str(e)}")
        return 0.0


@function
def cot(s, row):
    prompt = cot_prompt.format(question=row["question"])
    s += user(prompt)
    s += assistant(gen(name="cot", max_tokens=1024, temperature=0.7, top_p=0.9))


@function
def qa(s, row):
    prompt = doc_to_text.format(question=row["question"], cot=row["cot"])
    s += user(prompt)
    s += assistant(gen("model_output", max_tokens=512, temperature=0.7, top_p=0.9))
    s["score"] = check_answer_match(s["model_output"], row["answer"])


@function
def base_qa(s, row):
    prompt = "Question: {question}.".format(question=row["question"])
    s += user(prompt)
    s += assistant(gen("model_output", max_tokens=1024, temperature=0.7, top_p=0.9))
    # s["extract_output"] = extract_answer(s["model_output"])
    s["score"] = check_answer_match(s["model_output"], row["answer"])


def launch_engine():
    global save_dir
    if config["stage"] == 1:
        model_path = config["cot_model_path"]
    elif config["stage"] == 2:
        model_path = config["small_model_path"]
    elif config["stage"] == 3:
        model_path = config["base_model_path"]
    else:
        raise ValueError("stage should be 1 or 2 or 3")
    backend = Runtime(
        model_path=model_path,
        dtype="bfloat16",
        dp_size=config["dp_size"],
    )
    model_name = model_path.strip().split("/")[-1]
    save_dir = os.path.join("results", model_name)
    os.makedirs(save_dir, exist_ok=True)

    set_default_backend(backend)
    print(f"Using {model_path} as backend")


def main():
    seed_everything(config["seed"])

    launch_engine()
    if config["stage"] == 1:
        # using our finetuned model to generate efficient cot
        cot_json_files = []
        completion_tokens = []

        ds = load_dataset("openai/gsm8k", name="main", split="test")
        ds = ds.map(map_function)
        num_batches = (len(ds) - 1) // config["batch_size"] + 1
        for batch in tqdm(range(num_batches)):
            start_idx = batch * config["batch_size"]
            end_idx = min((batch + 1) * config["batch_size"], len(ds))
            states = cot.run_batch([{"row": ds[i]} for i in range(start_idx, end_idx)])
            for i in range(start_idx, end_idx):
                item = dict(ds[i])
                completion_tokens.append(
                    states[i - start_idx].get_meta_info("cot")["completion_tokens"]
                )
                item["cot"] = states[i - start_idx]["cot"]
                cot_json_files.append(item)
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        with open(f"data/{config['data_name']}_cot.json", "w") as f:
            json.dump(cot_json_files, f, indent=4, ensure_ascii=False)
        with open(f"results/{config['data_name']}_completion_length.json", "w") as f:
            json.dump(
                {"average_completion_tokens": np.mean(completion_tokens)},
                f,
                indent=4,
                ensure_ascii=False,
            )

    elif config["stage"] == 2:
        result_files = []
        ds = load_dataset("json", data_files=f"data/{config['data_name']}_cot.json")[
            "train"
        ]

        num_batches = (len(ds) - 1) // config["batch_size"] + 1
        for batch in tqdm(range(num_batches)):
            start_idx = batch * config["batch_size"]
            end_idx = min((batch + 1) * config["batch_size"], len(ds))
            states = qa.run_batch([{"row": ds[i]} for i in range(start_idx, end_idx)])
            for i in range(start_idx, end_idx):
                item = dict(ds[i])
                item["model_output"] = states[i - start_idx]["model_output"]
                item["score"] = states[i - start_idx]["score"]
                result_files.append(item)
        with open(f"results/{config['data_name']}_result.json", "w") as f:
            json.dump(result_files, f, indent=4, ensure_ascii=False)

        # compute accuracy
        accuracy = np.mean([item["score"] for item in result_files])
        # em_score = em.compute(
        #     references=ds["answer"],
        #     predictions=[item["extract_output"] for item in result_files],
        #     ignore_case=True,
        #     ignore_punctuation=True,
        # )["exact_match"]
        # invalid_count = sum(item["extract_output"] == INVALID for item in result_files)

        with open("results/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": accuracy,
                    "total_questions": len(ds),
                },
                fp=f,
                indent=4,
                ensure_ascii=False,
            )

    elif config["stage"] == 3:
        average_accuracy = []
        average_completion_tokens = []

        for n in range(config["n_repeat"]):
            print(f"Running {n+1}th repeat")
            result_files = []
            completion_tokens = []
            ds = load_dataset("openai/gsm8k", name="main", split="test")
            ds = ds.map(map_function)

            num_batches = (len(ds) - 1) // config["batch_size"] + 1
            for batch in tqdm(range(num_batches)):
                start_idx = batch * config["batch_size"]
                end_idx = min((batch + 1) * config["batch_size"], len(ds))
                states = base_qa.run_batch(
                    [{"row": ds[i]} for i in range(start_idx, end_idx)]
                )
                for i in range(start_idx, end_idx):
                    item = dict(ds[i])
                    item["model_output"] = states[i - start_idx]["model_output"]
                    item["score"] = states[i - start_idx]["score"]
                    completion_tokens.append(
                        states[i - start_idx].get_meta_info("model_output")[
                            "completion_tokens"
                        ]
                    )
                    result_files.append(item)

            with open(os.path.join(save_dir, f"{n}_result.json"), "w") as f:
                json.dump(result_files, f, indent=4, ensure_ascii=False)

            accuracy = np.mean([item["score"] for item in result_files])
            completion_tokens = np.mean(completion_tokens)

            average_accuracy.append(accuracy)
            average_completion_tokens.append(completion_tokens)
            print(f"{accuracy=}, {completion_tokens=}")

            with open(os.path.join(save_dir, f"{n}_metrics.json"), "w") as f:
                json.dump(
                    {
                        "accuracy": accuracy,
                        "total_questions": len(ds),
                        "average_completion_tokens": completion_tokens,
                    },
                    fp=f,
                    indent=4,
                    ensure_ascii=False,
                )
        with open(os.path.join(save_dir, "final_metrics.json"), "w") as f:
            json.dump(
                {
                    "average_accuracy": np.mean(average_accuracy),
                    "average_accuracy_std": np.std(average_accuracy),
                    "average_completion_tokens": np.mean(average_completion_tokens),
                    "average_completion_tokens_std": np.std(average_completion_tokens),
                },
                f,
                indent=4,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
