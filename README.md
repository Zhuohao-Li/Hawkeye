# Hawkeye: Model Collaboration for Efficient Reasoning (COLM'25)


This is the repository for the **Efficient_CoT** project.  
It aims to provide efficient and structured implementations for chain-of-thought reasoning in AI models.

## Evaluation Task (Accuracy/Responese Length/Throughtput)


| Task              | Model                                      | Accuracy(%) | Response Length | Throughput |
|------------------|--------------------------------------------|----------|----------------|------------|
| GSM8K           | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    |     85.65 $\pm$ 0.63  |      477.98 $\pm$ 0.89(Tokens)          |          |
|                | Efficient CoT                              |     82.11 $\pm$ 0.48    |   413.42 $\pm$ 2.19(Tokens)        |            |
| MATH-500       | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    |          |                |            |
|                | Efficient CoT                              |          |                |            |
| MQA            | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    |          |                |            |
|                | Efficient CoT                              |          |                |            |
| GPQA Diamond   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    |   38.72 $\pm$ 3.56 |        1975.19 $\pm$ 8.90 (Tokens)        |            |
|                | Efficient CoT                              |  39.23 $\pm$ 3.10    |       2006.30 $\pm$ 2.23 (Tokens)        |            |
| AIME 2024      | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    |          |                |            |
|                | Efficient CoT                                    |          |                |            |
| MATH           | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    |     91.47     |         751.5(Tokens)       |            |
|                | Efficient CoT                                  |     87.45     |       208.33(Tokens)         |            |



## Features

- 🧠 Focused on efficiency and scalability.
- 📊 Benchmarking multiple state-of-the-art models.
- 🛠️ Modular and extensible design for evaluation pipelines.

---

## Progress Tracker

### Model Evaluations

- [x] Qwen evaluation  
- [ ] Llama 3 evaluation *(in progress)*  

### Tasks

- [x] Implement data preprocessing  
- [ ] Optimize inference speed *(coming soon)*  

### Documentation

- [x] Initial README  
- [ ] Add user guide  

---

## How to Use

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Jianshu1only/Efficient_CoT.git

# Instruction Decoding benchmark

## GSM8K Subset (138 Questions, Simplified CoT : 30 tokens, Hint CoT : 10 tokens)

### Qwen-0.5B GSM8K Evaluation Results

This table summarizes the evaluation results of the Qwen-0.5B model on the GSM8K dataset under different configurations:

| Configuration              | Match Count | Total Count | Accuracy (%) |
|----------------------------|-------------|-------------|--------------|
| Baseline (no system prompt) | 38          | 138         | 27.54        |
| Baseline (with system prompt) | 45         | 138         | 32.61        |
| Instruction (no system prompt) | 82        | 138         | 59.42        |
| Instruction (with system prompt) | 86       | 138         | 62.32        |
| Instruction (with system prompt/ simplified) | 68       | 138         | 49.28       |
| Instruction (with system prompt/ hint) | 50      | 138         | 36.24        |

---

### Llama3-1B GSM8K Evaluation Results

| Configuration              | Match Count | Total Count | Accuracy (%) |
|----------------------------|-------------|-------------|--------------|
| Baseline (with system prompt) | 4         | 138         | 2.90       |
| Instruction (with system prompt) | 63       | 138         | 45.65        |
| Instruction (with system prompt/ simplified) | 55      | 138         | 39.86      |

---

### Compressed CoT

![Performance Chart](output.png)
