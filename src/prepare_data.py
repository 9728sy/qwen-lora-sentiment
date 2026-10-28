import os
import json
import random

def main():
    print("开始生成本地训练和测试数据...")
    
    # 手动构造一些简单的情感数据（正面和负面）
    positive_texts = [
        "这件衣服质量很好，穿起来很舒服。",
        "物流很快，包装完好，非常满意。",
        "老板态度很好，下次还会再来。",
        "物超所值，推荐大家购买。",
        "效果出乎意料的好，很喜欢。",
        "颜色很正，大小也合适，好评。",
        "非常满意的一次购物体验。",
        "用了一周，感觉很不错。"
    ]
    
    negative_texts = [
        "质量太差了，洗了一次就褪色。",
        "发货太慢，客服也不理人，差评。",
        "完全和图片不一样，退货了。",
        "用了两天就坏了，千万别买。",
        "味道很大，非常失望。",
        "尺码太小了，根本穿不进去。",
        "简直是浪费钱，太垃圾了。",
        "刚收到货就发现有瑕疵，无语。"
    ]

    # 把数据扩充一下，凑够需要的数量
    all_data = []
    for _ in range(100):
        for text in positive_texts:
            all_data.append({"text": text, "label": "正面"})
        for text in negative_texts:
            all_data.append({"text": text, "label": "负面"})

    # 打乱数据
    random.seed(42)
    random.shuffle(all_data)

    # 划分训练集(800条)和测试集(300条)
    train_data = all_data[:800]
    test_data = all_data[800:1100]
    
    def format_data(data):
        formatted = []
        for item in data:
            # 构造大模型微调需要的对话格式
            formatted_item = {
                "instruction": "请判断以下文本的情感倾向，只回答'正面'或'负面'。",
                "input": item["text"],
                "output": item["label"]
            }
            formatted.append(formatted_item)
        return formatted

    train_formatted = format_data(train_data)
    test_formatted = format_data(test_data)

    # 保存为 json 文件，放在 data 文件夹下
    os.makedirs("data", exist_ok=True)
    
    with open("data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_formatted, f, ensure_ascii=False, indent=2)
        
    with open("data/test.json", "w", encoding="utf-8") as f:
        json.dump(test_formatted, f, ensure_ascii=False, indent=2)
        
    print(f"数据处理完成！训练集：{len(train_formatted)} 条，测试集：{len(test_formatted)} 条。")
    print("文件已保存在 data/ 目录下。")

if __name__ == "__main__":
    main()