#!/usr/bin/env bash

model_name_or_paths=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

for model_name_or_path in "${model_name_or_paths[@]}"; do
    echo "Evaluating with CoT enabled for model: ${model_name_or_path}"
    python gsm8k_evaluate_cot.py --cot \
        --model_name_or_path ${model_name_or_path}
done

for model_name_or_path in "${model_name_or_paths[@]}"; do
    echo "Evaluating without CoT for model: ${model_name_or_path}"
    python gsm8k_evaluate_cot.py \
        --model_name_or_path "${model_name_or_path}"
done


# for model_name_or_path in "${model_name_or_paths[@]}"; do
#     echo "Evaluating with CoT enabled for model: ${model_name_or_path}"
#     python math_evaluate_cot.py --cot \
#         --model_name_or_path "${model_name_or_path}"
# done

# for model_name_or_path in "${model_name_or_paths[@]}"; do
#     echo "Evaluating without CoT for model: ${model_name_or_path}"
#     python math_evaluate_cot.py \
#         --model_name_or_path "${model_name_or_path}"
# done
