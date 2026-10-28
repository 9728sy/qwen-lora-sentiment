# Qwen2.5-0.5B LoRA 中文情感分类小项目

## 1. 任务说明
用公开中文情感分类数据（ChnSentiCorp），对 Qwen2.5-0.5B-Instruct 做 LoRA 微调。
输入：一条中文短文本。输出：正面 / 负面。

## 2. 测试集与指标
测试集：固定抽取 300 条，不参与训练。
指标：Accuracy、Macro-F1。

## 3. Baseline
Qwen2.5-0.5B-Instruct 原始模型，不微调，使用固定 Prompt 进行 Zero-shot 预测。

## 4. 方法
使用 PEFT 库对模型进行 LoRA 微调。

## 5. 结果对比
（待填）

## 6. 错误分析
（待填）

## 7. 如何运行
（待填）