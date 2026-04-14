import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import time

# 导入你之前写好的模块
from model import MiniChineseGPT, GPTConfig
from dataloader import MyTokenizer

# --- 1. SFT 数据集定义 ---
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, block_size):
        self.data = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 逐行读取 JSONL 数据
        print(f"正在加载 SFT 数据: {jsonl_path}...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                # 1. 拼接 Prompt 模板
                prompt_text = f"User: {item['input']}\nAssistant: "
                response_text = f"{item['target']}<|endoftext|>"
                
                # 2. 转化为 Token IDs
                prompt_ids = self.tokenizer.encode(prompt_text)
                response_ids = self.tokenizer.encode(response_text)
                
                # 3. 拼接完整的输入序列
                full_ids = prompt_ids + response_ids
                
                # 4. 核心：构造 Labels (Loss Masking)
                # 提问部分的 label 设为 -100，回答部分的 label 保持不变
                labels = [-100] * len(prompt_ids) + response_ids
                
                # 5. 截断防止超长
                full_ids = full_ids[:block_size+1]
                labels = labels[:block_size+1]
                
                # 如果总长度至少为2（能切分出 x 和 y），则加入数据集
                if len(full_ids) >= 2:
                    self.data.append((full_ids, labels))
        
        print(f"成功加载了 {len(self.data)} 条 SFT 问答对！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_ids, labels = self.data[idx]
        
        # 错位截取 X 和 Y
        x = torch.tensor(full_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y

# --- 2. 动态填充函数 (Collate Fn) ---
# 因为每句问答的长度都不一样，我们需要把同一个 Batch 里的句子补齐到相同长度
def collate_fn(batch):
    xs, ys = zip(*batch)
    # 用 0 填充输入 x，用 -100 填充目标 y (让填充部分也不计算 Loss)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=-100)
    return x_padded, y_padded

# --- 3. 训练主流程 ---
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRETRAINED_WEIGHTS = "mini_llm_epoch_9.pt" # 换成你预训练保存的最新权重
    MERGES_PATH = "my_tokenizer_merges.json"
    
    # 1. 初始化分词器和数据集
    tokenizer = MyTokenizer(MERGES_PATH)
    # 注意：你需要先创建一个 dummy_sft.jsonl 来测试，或者下载真实的 SFT 数据
    dataset = SFTDataset("sft_data.jsonl", tokenizer, block_size=256) 
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # 2. 加载预训练好的模型
    config = GPTConfig()
    model = MiniChineseGPT(config)
    print(f"正在加载预训练大脑: {PRETRAINED_WEIGHTS}...")
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)
    
    # SFT 的学习率通常比预训练小很多（比如预训练 3e-4，微调 2e-5）
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    # 3. 开始 SFT 训练
    print("\n🚀 SFT 微调开始！教模型做人...")
    model.train()
    
    max_epochs = 3 # SFT 不需要跑太多 Epoch，通常 1-3 轮就足够收敛了
    for epoch in range(max_epochs):
        for step, (X, Y) in enumerate(dataloader):
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            
            optimizer.zero_grad()
            logits, loss = model(X, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                
        # 保存微调后的权重
        torch.save(model.state_dict(), f"mini_llm_sft_epoch_{epoch}.pt")
        print(f"💾 SFT Epoch {epoch} 权重已保存！")

if __name__ == "__main__":
    main()