import torch
from torch.nn import functional as F
from sft.model import MiniChineseGPT, GPTConfig
from sft.dataloader import MyTokenizer

# --- 1. 参数与路径配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_PATH = "mini_llm_epoch_9.pt" # 替换为你实际训练出来的最新权重文件名
MERGES_PATH = "my_tokenizer_merges.json" # 你的词汇表规则

# --- 2. 加载大脑 (分词器 + 模型) ---
print("正在加载分词器...")
tokenizer = MyTokenizer(MERGES_PATH)

print("正在构建模型架构...")
config = GPTConfig()
model = MiniChineseGPT(config)

print(f"正在注入灵魂 (加载权重: {WEIGHT_PATH})...")
try:
    # 加载训练好的权重
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # 极其重要：将模型切换到推理模式（关闭 Dropout 等训练专属机制）
    print("加载成功！模型已准备就绪。\n")
except FileNotFoundError:
    print(f"❌ 找不到权重文件 {WEIGHT_PATH}，请确保你已经先运行过 train.py 并保存了模型。")
    exit()

# --- 3. 核心生成算法 (带温度与 Top-p 采样) ---
@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    # 1. 文本转 Token ID
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        # 截断上下文
        idx_cond = idx[:, -config.block_size:]
        
        # 获取预测 logits
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] # (Batch=1, Vocab_size)
        
        # --- 魔法时刻：Temperature + Top-P 采样 ---
        
        # 1. 应用温度缩放
        logits = logits / temperature
        
        if top_p is not None and top_p < 1.0:
            # 2. 将 logits 降序排序
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            # 3. 计算排序后的累积概率 (Softmax 然后 Cumsum)
            # 注意：必须先做 softmax 变成概率，才能算累加和
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 4. 找到那些累积概率超过 top_p 的位置，创建一个布尔掩码
            # 将超过阈值的部分设为 True (我们要剔除的部分)
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # 5. 关键细节：向右平移一位
            # 为了保证就算所有概率都很平，我们至少保留一个最可能的词
            # 我们把掩码向右平移一位，强制保留第一个超过阈值的词
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 6. 把原本词表位置上的对应 Token 标记为 -无穷大
            # scatter_ 讲人话就是：按照 sorted_indices 的指引，把 True 的地方在原 logits 里填满 -inf
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
        # 7. 再次 Softmax，把没被砍掉的词重新变成概率分布（加起来等于 1）
        probs = F.softmax(logits, dim=-1)
        
        # 8. 抽卡采样
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 9. 拼接到输入中
        idx = torch.cat((idx, idx_next), dim=1)

    # 循环结束，解码 (用你之前写好的解码逻辑或假装解码的打印)
    generated_ids = idx[0].tolist()
    print(f"底层 Token 序列 (Top-P 模式): {generated_ids}")

    # 解码回文本
    output = tokenizer.decode(generated_ids)
    return output

# --- 4. 互动测试 ---
if __name__ == "__main__":
    print("="*40)
    print("🤖 Mini-LLM 终端聊天室")
    print("="*40)
    
    while True:
        user_input = input("\n请输入开头 (输入 'q' 退出) > ")
        if user_input.lower() == 'q':
            break
            
        print("\n模型思考中...")
        # 调用生成函数
        # 参数调整建议：
        # - 如果模型总是说重复的话，提高 temperature (如 1.2)
        # - 如果模型乱码、不知所云，降低 temperature (如 0.5)
        output = generate_text(model, tokenizer, user_input, max_new_tokens=50, temperature=0.8, top_p=0.9)
        
        print("\n[生成结果]:")
        print(output)