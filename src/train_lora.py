import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# 设置模型名称
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    print("1. 正在加载 Tokenizer 和 模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # 使用 bfloat16 节省显存
        device_map="auto"
    )
    
    # 2. 加载造好的本地数据
    print("2. 正在加载本地数据集...")
    data_files = {"train": "data/train.json", "test": "data/test.json"}
    dataset = load_dataset("json", data_files=data_files)
    
    # 3. 数据处理函数
    def process_func(example):
        # 构造系统提示词和用户输入
        prompt = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
        response = f"{example['output']}<|im_end|>"
        
        # 把 prompt 和 output 拼起来
        full_text = prompt + response
        
        # 转换为 token
        tokenized = tokenizer(full_text, max_length=256, truncation=True, padding=False)
        
        # 把 prompt 部分的 token 设为 -100，只学习 response 部分
        prompt_tokenized = tokenizer(prompt, max_length=256, truncation=True)
        prompt_len = len(prompt_tokenized["input_ids"])
        
        labels = tokenized["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len
        
        tokenized["labels"] = labels
        return tokenized
    
    print("3. 正在处理数据集格式...")
    tokenized_datasets = dataset.map(process_func, remove_columns=dataset["train"].column_names)
    
    # 4. 配置 LoRA
    print("4. 配置 LoRA 参数...")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8, # LoRA 秩
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    # 将原模型转换为 PEFT 模型（冻结原参数，只训练 LoRA 层）
    model = get_peft_model(model, config)
    model.print_trainable_parameters() # 打印出参数量，会发现只训练了极少一部分参数
    
    # 5. 配置训练参数
    args = TrainingArguments(
        output_dir="./output/Qwen_lora",
        per_device_train_batch_size=4, # 显存如果爆了，改成 2
        gradient_accumulation_steps=2,
        logging_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=False,
        report_to="none" # 不连接 wandb 等外部工具
    )
    
    # 6. 开始训练
    print("5. 开始微调训练...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    
    trainer.train()
    
    # 7. 保存微调后的 LoRA 权重
    print("6. 训练结束，保存模型...")
    trainer.save_model("./output/Qwen_lora_final")
    print("完美！模型已保存。")

if __name__ == "__main__":
    main()