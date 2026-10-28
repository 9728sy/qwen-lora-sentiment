import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
lora_path = "./output/Qwen_lora_final"

print("加载微调后的模型...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

def predict(text):
    prompt = f"<|im_start|>system\n请判断以下文本的情感倾向，只回答'正面'或'负面'。<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

with open("data/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# 为了快速看结果，这里只测前 100 条
test_data = test_data[:100]

print("开始评估...")
y_true = []
y_pred = []

for item in test_data:
    pred = predict(item['input'])
    if "正面" in pred: pred = "正面"
    elif "负面" in pred: pred = "负面"
    
    y_true.append(item['output'])
    y_pred.append(pred)

# 计算指标
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, pos_label="正面", average="macro")

print(f"\n✅ 评估完成！")
print(f"Accuracy (准确率): {acc:.4f}")
print(f"Macro-F1 分数: {f1:.4f}")