from datasets import load_dataset
from sglang import assistant, gen, set_default_backend, Runtime, user, function
from argparse import ArgumentParser
import numpy as np
import os
from collections import defaultdict

prompt_template = """
Question: {question}
"""

prompt_template_cot = """
Question: {question}
Chain_of_thought: {cot}
"""
INVALID = -9999999

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

# Above function are referenced by https://github.com/hendrycks/math/blob/main/modeling/eval_math_gpt.py

@function
def qa(s, p):
    s += user(p)
    s += assistant("Answer: " + gen(name="answer", max_tokens=args.max_new_tokens, temperature=0), stop=["Question", "Assistant:", "<|separator|>"])

    
def create_prompt(example, is_cot=False):
    if is_cot:
        return prompt_template_cot.format(question=example["problem"], cot=example["cot"])
    return prompt_template.format(question=example["problem"])


def get_answer_value(answer_str):
    idx = answer_str.rfind("\\boxed")
    if idx < 0:
        idx = answer_str.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(answer_str):
        if answer_str[i] == "{":
            num_left_braces_open += 1
        if answer_str[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = answer_str[idx:right_brace_idx + 1]

    left = "\\boxed{"
    try:
        assert retval[:len(left)] == left
        assert retval[-1] == "}"
        return retval[len(left):-1]
    except:
        return INVALID 


def parse_args():
    parser = ArgumentParser(description="Parse arguments for testing")
    
    # Model and data
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained model or model identifier from HuggingFace Model Hub.")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset to be used (e.g., GSM8K).")
    
    # Chain of Thought (CoT) configuration
    parser.add_argument("--cot", action="store_true", help="Enable Chain of Thought reasoning for the model.")
    
    # Inference configurations
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate.")
    
    return parser.parse_args()

def main():
    ds = load_dataset("lighteval/MATH", name="all", split="test")

    runtime = Runtime(model_path=args.model_name_or_path, tp_size=1)
    set_default_backend(backend=runtime)

    num_batches = (len(ds) - 1) // args.batch_size + 1

    # Dict: (problem_level, problem_type) => [True, False] a List of results
    cors = defaultdict(list)

    # Dict: problem_type => [True, False] a List of results
    subject_cors = defaultdict(list)

    # Dict: level => [True, False] a List of results
    level_cors = defaultdict(list)

    preds = []
    subjects = []
    levels = []
    labels = []

    for batch in range(num_batches):
        start = batch * args.batch_size
        end = min((batch + 1) * args.batch_size, len(ds)) 
        states = qa.run_batch([{
            "p": create_prompt(example=ds[i], is_cot=args.cot)
        } for i in range(start, end)])
        preds.extend(get_answer_value(s["answer"]) for s in states)
        labels.extend(get_answer_value(ds[i]["solution"]) for i in range(start, end))
        subjects.extend(ds[i]["type"] for i in range(start, end))
        levels.extend(int(ds[i]["level"].split("Level ")[1]) for i in range(start, end))
        
    correct = 0
    total = len(preds)
    subject_names = sorted(subject_cors.keys())

    # compute
    for pred, label, subject, level in zip(preds, labels, subjects, levels):
        equiv = is_equiv(pred, label)
        if equiv: correct += 1
        cors[(level, subject)].append(equiv)
        level_cors[level].append(equiv)
        subject_cors[subject].append(equiv)
        

    # create dir containing the model name to store the results
    model_name_or_path = args.model_name_or_path.strip("/ ")
    model_name = model_name_or_path.split("/")[-1]
    output_path = f"results/{model_name}"
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "MATH_result.txt"), "w+") as f:
        for k, (pred, label, subject, level) in enumerate(zip(preds, labels, subject_names, levels)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, subject, level, pred, label))

        # print(cors)
        for subject in subject_names:
            for level in [1, 2, 3, 4, 5]:
                if (level, subject) in cors:
                    cors_list = cors[(level, subject)]
                    print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                    f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))

        print("#####################")
        f.write("#####################\n")
        # also get accuracies for each 
        for level in sorted(level_cors):
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")

        for subject in subject_names:
            # for subject in sorted(subject_cors):
            if subject in subject_cors:
                cors_list = subject_cors[subject]
                print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total))

    runtime.shutdown()

if __name__ == "__main__":
    args = parse_args()
    main()


