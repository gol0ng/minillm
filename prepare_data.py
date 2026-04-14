from datasets import load_dataset

print("开始下载新版维基百科中文数据...")

dataset = load_dataset(
    "wikimedia/wikipedia", 
    "20231101.zh", 
    split="train", 
    streaming=True
)

count = 0
with open("../datasets/corpus.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        text = item['text']
        if len(text) > 100: 
            f.write(text + "\n\n")
            count += 1
            if count % 10000 == 0:
                print(f"已处理 {count} 篇文章...")
        
        if count >= 100000: 
            break

print("数据收集完成！你的 corpus.txt 准备好了。")