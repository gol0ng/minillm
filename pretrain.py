import torch
import torch.optim as optim
import time
from model import MiniChineseGPT, GPTConfig
from dataloader import WikiDataset, MyTokenizer
from torch.utils.data import DataLoader

# --- 1. 硬件设备配置 (GPU 加速) ---
# 深度学习的心跳，必须在显卡上跑
device = "cuda" if torch.cuda.is_available() else "cpu"
# 如果你用的是 Mac 的 M1/M2/M3 芯片，可以用 mps 加速：
# device = "mps" if torch.backends.mps.is_available() else device
print(f"当前使用的计算设备: {device}")

# --- 2. 组装 DataLoader (燃油管道) ---
print("正在加载分词器和数据集...")
tokenizer = MyTokenizer("my_tokenizer_merges.json")
dataset = WikiDataset("corpus.txt", tokenizer, block_size=256, max_chars=None)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- 3. 点火：初始化模型 ---
config = GPTConfig() # 这里面有你设定的参数：n_embd=512, n_head=8, n_layer=4 等
model = MiniChineseGPT(config)
model.to(device) # 把模型搬到显卡上
print(f"模型已加载到 {device}，总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# --- 4. 优化器配置 (驾驶员) ---
# AdamW 是目前训练 Transformer 的绝对金标准
# lr=3e-4 是一个小模型非常经典的起始学习率
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# --- 5. 训练主循环 (跑圈开始！) ---
max_epochs = 10 
log_interval = 100 # 每跑100步打印一次状态

print("\n🚀 引擎点火，开始训练...")
model.train() # 将模型切换到训练模式

for epoch in range(max_epochs):
    epoch_start_time = time.time()
    
    for step, (X, Y) in enumerate(dataloader):
        # 将数据搬到显卡上
        X, Y = X.to(device), Y.to(device)
        
        # 1. 前向传播：模型阅读 X，预测下一个字，并计算与 Y 的误差 (Loss)
        logits, loss = model(X, Y)
        
        # 2. 清空上一步的残余梯度
        optimizer.zero_grad(set_to_none=True)
        
        # 3. 反向传播：计算误差对每个参数的导数 (求导)
        loss.backward()
        
        # 4. 梯度裁剪：防止 Transformer 训练早期梯度爆炸（极其重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 5. 更新权重：根据导数微调模型的大脑
        optimizer.step()
        
        # --- 打印训练日志 ---
        if step % log_interval == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
            
    # 每一个 Epoch 结束后，保存一次模型权重 (存档)
    checkpoint_path = f"mini_llm_epoch_{epoch}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 Epoch {epoch} 完成！耗时 {time.time() - epoch_start_time:.2f}s。模型已保存至 {checkpoint_path}")

print("🎉 恭喜！模型训练全部完成！")