## Generate the chain of Thought

code: `gen_cot.py`

prompt template:

```python
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
```

Core Logic:

1. **Utilize GPT-4o**:
   - GPT-4o is an exceptional and highly capable model that thoroughly understands my intentions.
   - It generates precise and actionable content, making it an excellent choice for this task.
2. **Store `cot` Output**:
   - Save the generated chain of thought (`cot`) into a JSON Lines file named `gsm8k_cot.jsonl` for further analysis and evaluation.



## Benchmark: Two-Turn Conversation

1. **First Turn**:
   - **Objective**: Prompt the model to "think" and provide a reasoning process along with an answer.
   - **Interaction**: Ask the model to reason through the problem and present its thought process.
2. **Second Turn**:
   - **Objective**: Directly ask the model for the final answer without additional reasoning.
   - **Interaction**: Request the model to provide the answer explicitly for extraction and evaluation.

this is an example about above interaction from Llama-3.1-8B-Instruct

```txt
<|start_header_id|>user<|end_header_id|>


Below is a question followed by a chain of thought. Please answer the question based on the reasoning provided in the chain of thought.

Question: John runs 60 miles a week. He runs 3 days a week.  He runs 3 hours the first day and half as much the other two days he runs.  How fast does he run?

Chain of Thought: 1. First, identify how much time John spends running in total each week by summing his running time over the three days.
2. Understand that John runs 60 miles a week over the course of these runs.
3. On the first day, John runs for 3 hours. Determine how many miles he runs on this first day by setting it as an unknown variable and considering his pace.
4. For the other two days, note that John runs half as much as he does on the first day, which means he runs for 1.5 hours on each of those days.
5. Express the total distance covered over the week in terms of his running speed in miles per hour (mph).
6. Establish an equation where the sum of the runs for all three days equals 60 miles.
7. The equation would have John's running distance for the first day as 3 times his speed and 1.5 times his speed for each of the other two days.
8. Solve this equation to determine the running speed that equates to the total weekly distance of 60 miles.
9. Cross-check the calculated running speed by breaking down the total distance into daily distances and confirming these match with 60 miles overall.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

## Step 1: Calculate the total time John spends running each week.
John runs 3 hours the first day and half as much (1.5 hours) on each of the other two days, so the total time spent running is 3 + 1.5 + 1.5 = 6 hours.

## Step 2: Understand that John runs 60 miles a week over the course of these runs.
This is given in the problem.

## Step 3: Determine how many miles John runs on the first day.
Let's denote John's running speed as "s" miles per hour. On the first day, he runs for 3 hours, so the distance covered on the first day is 3s miles.

## Step 4: Calculate the distance covered on the other two days.
On each of the other two days, John runs for 1.5 hours, so the distance covered on each of these days is 1.5s miles.

## Step 5: Express the total distance covered over the week in terms of John's running speed.
The total distance covered is the sum of the distances covered on each day, which is 3s + 1.5s + 1.5s = 6s miles.

## Step 6: Establish an equation where the sum of the runs for all three days equals 60 miles.
Since the total distance covered is 60 miles, we can set up the equation 6s = 60.

## Step 7: Solve the equation to determine John's running speed.
To solve for s, we divide both sides of the equation by 6, giving us s = 60 / 6 = 10.

## Step 8: Cross-check the calculated running speed by breaking down the total distance into daily distances.
On the first day, John runs 3s = 3 * 10 = 30 miles. On each of the other two days, he runs 1.5s = 1.5 * 10 = 15 miles. The total distance covered is 30 + 15 + 15 = 60 miles, which matches the given total weekly distance.

The final answer is: $\boxed{10}$<|eot_id|><|start_header_id|>user<|end_header_id|>

Please provide the answer directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer: 10<|eot_id|>Simplify the equation by combining like terms and solving for 'x'. Clarify what each term in the equation represents: '80 + x' is the total number of Post-it notes Candice had after the purchase, and '220' is the number of notes used, setting up the equation: '80 + x - 220 = 23'.
7. Rearrange and simplify the equation: '80 + x = 220 + 23'.
8. Simplifying further: '80 + x = 243'.
9. Solve for 'x' by subtracting 80 from both sides of the equation to isolate 'x': 'x = 243 - 80'.
10. Perform the subtraction to find the value of 'x', which represents the number of Post-it notes in the package Candice purchased. This will yield the solution indicating the number of notes in the purchased package.

x = 243 - 80
x = 163

Therefore, Candice purchased a package of 163 Post-it notes.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please provide the answer directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer: 163<|eot_id|> dragon's flame reach while still being able to hit the dragon with the gold javelin.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please provide the answer directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer: 200 feet.<|eot_id|>
```



### GSM8K

- [x] CoT generation
- [x] benchmark code 

code: `gsm8k_evaluate_cot.py`

The script stores the results for each model in the `results/` directory. Each directory contains six files:

1. **`example(_cot).txt`:**
   - Contains the example interactions.
2. **`gsm8k_args(_cot).json`**:
   - Includes the experimental arguments used in the evaluation.
3. **`gsm8k_result(_cot).json`**:
   - Stores the calculated metrics from the evaluation.



**results:**

| Model                 | CoT (Accuracy)     | No-CoT (Accuracy)  | Difference (CoT - No-CoT) |
|-----------------------|--------------------|--------------------|---------------------------|
| Qwen2.5-7B-Instruct   | 0.8893100833965125 | 0.8915845337376801 | -0.0022744503411676       |
| Qwen2.5-3B-Instruct   | 0.8438210765731615 | 0.7717968157695224 |  0.0720242608036391       |
| Qwen2.5-1.5B-Instruct | 0.8385140257771039 | 0.7210007581501138 |  0.1175132676269901       |
| Qwen2.5-0.5B-Instruct | 0.7043214556482184 | 0.47763457164518575|  0.2266868840030326       |
| Llama-3.1-8B-Instruct | 0.9052312357846853 | 0.8589840788476119 |  0.0462471569370734       |
| Llama-3.2-3B-Instruct | 0.890068233510235  | 0.7740712661106899 |  0.1159969673995451       |
| Llama-3.2-1B-Instruct | 0.7513267626990144 | 0.5519332827899924 |  0.1993934799090219       |




### MATH

- [ ] **CoT Generation**  
  - Code is complete, but I haven't run it yet due to insufficient API keys. 😢  

- [x] **Benchmark Code**  
  - Code is implemented but hasn't been thoroughly tested.  