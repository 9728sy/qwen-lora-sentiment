from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 指定模型名称
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 2. 加载 Tokenizer 和 模型
print("正在下载和加载模型，请稍等...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 3. 准备测试文本
text = "这件衣服质量太差了，穿了一天就起球，千万别买！"

# 4. 构建 Prompt
messages = [
    {"role": "system", "content": "你是一个情感分析助手。请判断用户输入文本的情感倾向，只回答'正面'或'负面'。"},
    {"role": "user", "content": text}
]

# 5. 让模型生成回答
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

print("模型正在思考...")
outputs = model.generate(**inputs, max_new_tokens=10)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(f"\n输入文本：{text}")
print(f"模型判断：{response}")