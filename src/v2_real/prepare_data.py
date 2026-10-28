import pandas as pd
import json
import os

def main():
    print("开始处理真实数据集...")
    # 下载的 parquet 文件
    df = pd.read_parquet("data/train-00000-of-00001.parquet")
    print("数据列名:", df.columns.tolist())

    # 兼容不同的列名（通常文本叫 'text'，标签叫 'label'）
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]

    # 打乱顺序并抽取数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df.iloc[:2000]       # 2000 条训练
    test_df = df.iloc[2000:2500]    # 500 条测试

    def format_data(data_df):
        formatted = []
        for _, row in data_df.iterrows():
            text = str(row[text_col])
            label = int(row[label_col])
            # ChnSentiCorp 通常 1 是正面，0 是负面
            answer = "正面" if label == 1 else "负面"
            formatted.append({
                "instruction": "请判断以下文本的情感倾向，只回答'正面'或'负面'。",
                "input": text,
                "output": answer
            })
        return formatted

    os.makedirs("data", exist_ok=True)
    with open("data/train.json", "w", encoding="utf-8") as f:
        json.dump(format_data(train_df), f, ensure_ascii=False, indent=2)
    with open("data/test.json", "w", encoding="utf-8") as f:
        json.dump(format_data(test_df), f, ensure_ascii=False, indent=2)

    print(f"数据处理完成！训练集：{len(train_df)} 条，测试集：{len(test_df)} 条。")

if __name__ == "__main__":
    main()