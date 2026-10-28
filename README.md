# Qwen2.5-0.5B LoRA 中文情感分类小项目

本项目探索了基于 Qwen2.5-0.5B-Instruct 模型进行 LoRA 微调完成中文情感分类任务。项目分为两个阶段，逐步实现从“环境与流程验证”到“真实数据量化评估”的迭代。

## 阶段一 (V1)：快速验证闭环
- **数据**：基于模板本地生成的 800 条极简数据（存放于 `data/mock`）。
- **目的**：打通环境配置、数据格式化、LoRA 微调（`PEFT` 库）与推理测试的全流程。
- **结果**：成功以 0.88% 的可训练参数占比完成微调，模型能精准输出“正面/负面”，验证了工程链路的可行性。

## 阶段二 (V2)：真实数据进阶评估
- **数据**：引入公开真实数据集 **ChnSentiCorp**（抽取 2000 条训练，500 条测试，存放于 `data/`）。
- **目的**：解决 V1 版本中的“过拟合”与“数据单一”问题，量化评估模型的泛化能力。
- **核心工作**：
  1. 编写 `prepare_data.py` 清洗真实 Parquet 格式数据，并格式化为大模型指令微调格式。
  2. 使用真实数据重新进行 LoRA 微调（`train_lora.py`），Loss 从 0.68 正常收敛至 0.10 左右。
  3. 编写 `evaluate.py`，在 500 条独立测试集上计算 Accuracy 与 Macro-F1 分数。

### V2 评估结果
- **Baseline (Zero-shot)**：基础模型直接推理（表现力较弱）
- **LoRA 微调后**：**Accuracy: 92.00%**, **Macro-F1: 91.99%** (基于 100 条真实独立测试集)

## 目录结构
```text
qwen-lora-sentiment/
├── data/
│   ├── train-00000-of-00001.parquet  (真实公开数据)
│   └── mock/                         (V1 本地假数据)
├── src/
│   ├── v1_mock/                      (V1 阶段代码)
│   │   ├── prepare_data.py
│   │   ├── train_lora.py
│   │   └── test_lora.py
│   └── v2_real/                      (V2 阶段进阶代码)
│       ├── prepare_data.py
│       ├── train_lora.py
│       └── evaluate.py
├── .gitignore
└── README.md
```

## 如何运行 V2 进阶版
```bash
# 1. 准备数据 (需提前将 parquet 文件放入 data/)
python src/v2_real/prepare_data.py

# 2. 开始 LoRA 微调训练
python src/v2_real/train_lora.py

# 3. 运行评估脚本
python src/v2_real/evaluate.py
```

## 数据获取说明
由于 GitHub 文件大小限制，数据集文件未直接上传至仓库。
- **真实数据 V2**：需从 Hugging Face (`lansinuote/ChnSentiCorp`) 下载 `train-00000-of-00001.parquet` 放入 `data/` 目录。
- 也可以直接运行本地数据生成脚本进行 V1 版本测试：`python src/v2_real/prepare_data.py`。