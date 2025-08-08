# 使用 HuggingFace 模型进行评估
```
python evaluation.py --model_path Qwen/Qwen1.5-7B --test_file test.jsonl --eval_mode joint
```
# 使用本地模型进行评估
```
python evaluation.py --model_path ./checkpoints/checkpoint-50 --test_file ./dataset/test.jsonl --eval_mode single --is_local
```
```
--model_path : 测试的模型，如果为本地模型，需要加上 --is_local

--test_file  : 包括question/answer的文件

--eval_mode  : joint为大小模型联合推理， single为单独测试大模型
```

需要在脚本的77行添加api key：        
```
client = openai.OpenAI(api_key="")
```
