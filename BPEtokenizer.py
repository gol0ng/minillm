import json
# --- 1. 准备训练语料并转换为底层字节 ---
with open("../datasets/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read(50000)

# 将中文字符串编码为 UTF-8 字节，并转换为 0-255 的整数列表
# 这是所有现代 LLM 词表的绝对起点 (Vocab Size = 256)
tokens = list(text.encode("utf-8"))
# print(f"原始字节长度: {len(tokens)}")
# print(f"所有字节: {tokens}")

# --- 2. 核心算法：统计相邻对频率 ---
def get_stats(ids):
    counts = {}
    # 遍历所有相邻的两个 token
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    # print(f"当前相邻对频率: {counts}")
    return counts

# --- 3. 核心算法：合并最高频的对 ---
def merge(ids, pair, idx):
    """
    在 ids 列表中，将所有匹配的 pair 替换为新的 idx
    """
    newids = []
    i = 0
    while i < len(ids):
        # 检查是否匹配到了目标 pair
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2 # 匹配成功，跳过两个原始 token
        else:
            newids.append(ids[i])
            i += 1 # 没匹配上，保留原样
    return newids

# --- 4. 训练你的分词器！ ---
vocab_size = 8000 # 目标词表大小。基础字节有 256 个，我们训练 20 个新词合并
num_merges = vocab_size - 256

# 记录合并规则的字典 (这是分词器的心脏)
# 格式: {(token1, token2): new_token_id}
merges = {} 

for i in range(num_merges):
    stats = get_stats(tokens)
    if not stats:
        break
    
    # 找出频率最高的那一对
    best_pair = max(stats, key=stats.get)
    
    # 给这个新组合分配一个新的 ID (从 256 开始递增)
    new_id = 256 + i 
    
    # 执行合并
    tokens = merge(tokens, best_pair, new_id)
    
    # 记录规则
    merges[best_pair] = new_id
    # print(f"合并 {i+1}: {best_pair} -> {new_id} (出现次数: {stats[best_pair]})")


saveable_merges = {f"{k[0]},{k[1]}": v for k, v in merges.items()}

with open("my_tokenizer_merges.json", "w", encoding="utf-8") as f:
    json.dump(saveable_merges, f)

print("词汇表规则已成功保存！")