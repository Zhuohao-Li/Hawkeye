from datasets import load_dataset
from utils import process_docs
import json


ds = load_dataset("Idavidrein/gpqa", name="gpqa_diamond", split="train")
ds = process_docs(ds)
json_files = []
for item in ds:
    json_files.append(
        {
            "choice1": item["choice1"],
            "choice2": item["choice2"],
            "choice3": item["choice3"],
            "choice4": item["choice4"],
            "answer": item["answer"],
            "Question": item["Question"],
        }
    )
with open("gpqa_diamond.json", "w") as f:
    json.dump(json_files, f, indent=4, ensure_ascii=False)
