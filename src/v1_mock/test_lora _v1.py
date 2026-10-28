import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
lora_path = "./output/Qwen_lora_final"

print("1. 加载微调后的模型...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
# 挂载微调好的 LoRA 权重
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

def predict(text):
    prompt = f"<|im_start|>system\n请判断以下文本的情感倾向，只回答'正面'或'负面'。<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, eos_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# 测试几条没在训练集里的新句子
test_texts = [
    "包装得很好，里面的东西一点都没坏，下次还来买。",
    "用了一天就坏了，退货还极其麻烦，避雷这家店！",
    "感觉一般般吧，没有想象中那么好，但也不算差。",
    "质量杠杠的，同事们都说好看，我也很开心。"
]

print("2. 开始测试：\n" + "-"*30)
for text in test_texts:
    result = predict(text)
    print(f"输入：{text}\n模型回答：{result}\n")