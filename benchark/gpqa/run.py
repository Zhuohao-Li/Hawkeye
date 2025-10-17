from datasets import load_dataset
from utils import doc_to_text, cot_prompt, base_prompt_template
from sglang import assistant, set_default_backend, Runtime, function, user, gen, select
import yaml
import numpy as np
from tqdm import tqdm
import json
import os
from utils import seed_everything

with open("run.yaml", "r") as f:
    config = yaml.safe_load(f)

save_dir = ""


@function
def cot(s, row):
    prompt = cot_prompt.format(question=row["Question"])
    s += user(prompt)
    s += assistant(gen(name="cot", max_tokens=1024, temperature=0.7, top_p=0.9))


@function
def qa(s, row):
    prompt = doc_to_text.format(
        choice1=row["choice1"],
        choice2=row["choice2"],
        choice3=row["choice3"],
        choice4=row["choice4"],
        Question=row["Question"],
        cot=row["cot"],
    )
    s += user(prompt)
    s += assistant(
        select(
            name="model_answer", choices=["(A)", "(B)", "(C)", "(D)"], temperature=0.0
        )
    )


@function
def base_qa(s, row):
    prompt = base_prompt_template.format(
        Question=row["Question"],
        choice1=row["choice1"],
        choice2=row["choice2"],
        choice3=row["choice3"],
        choice4=row["choice4"],
    )
    s += user(prompt)
    s += assistant(
        gen("model_output", max_tokens=2048, temperature=0.7, top_p=0.9)
        + "So the answer is "
        + select(
            name="model_answer", choices=["(A)", "(B)", "(C)", "(D)"], temperature=0.0
        )
    )


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
        tp_size=config["tp_size"],
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
        # generate cot
        print("Stage 1: generate cot")
        cot_json_files = []
        completion_tokens = []
        ds = load_dataset("json", data_files="data/gpqa_diamond.json")["train"]
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
        with open(f"results/{config['data_name']}_tokens.json", "w") as f:
            json.dump(
                {"average_completion_tokens": np.mean(completion_tokens)},
                f,
                indent=4,
                ensure_ascii=False,
            )

    elif config["stage"] == 2:
        print("Stage 2: using small model to generate answer depending on cot")
        print(f"{config['batch_size']=}")
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
                item["model_answer"] = states[i - start_idx]["model_answer"]
                result_files.append(item)
        with open(f"results/{config['data_name']}_result.json", "w") as f:
            json.dump(result_files, f, indent=4, ensure_ascii=False)

        # compute accuracy
        all_counts = len(result_files)

        right_counts = sum(
            item["answer"] == item["model_answer"] for item in result_files
        )
        accuracy = right_counts / all_counts

        with open("results/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": accuracy,
                    "total_questions": all_counts,
                },
                fp=f,
                indent=4,
                ensure_ascii=False,
            )

    elif config["stage"] == 3:
        print("Stage 3: using base model to generate answer")
        average_accuracy = []
        average_completion_tokens = []

        for n in range(config["n_repeat"]):
            print("====================")
            print(f"Running {n+1}th repeat")
            print("====================")
            result_files = []
            completion_tokens = []
            ds = load_dataset("json", data_files=f"data/{config['data_name']}.json")[
                "train"
            ]
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
                    item["model_answer"] = states[i - start_idx]["model_answer"]
                    completion_tokens.append(
                        states[i - start_idx].get_meta_info("model_output")[
                            "completion_tokens"
                        ]
                    )
                    result_files.append(item)

            with open(os.path.join(save_dir, f"{n}_result.json"), "w") as f:
                json.dump(result_files, f, indent=4, ensure_ascii=False)

            # compute accuracy
            accuracy = np.mean(
                [item["answer"] == item["model_answer"] for item in result_files]
            )

            completion_tokens = np.mean(completion_tokens)
            average_accuracy.append(accuracy)
            average_completion_tokens.append(completion_tokens)
            print(f"{accuracy=}, {completion_tokens=}")

            with open(os.path.join(save_dir, f"{n}_metrics.json"), "w") as f:
                json.dump(
                    {
                        "accuracy": accuracy,
                        "total_questions": len(ds),
                        "completion_tokens": completion_tokens,
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
