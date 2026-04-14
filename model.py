import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 1. 模型超参数配置 ---
class GPTConfig:
    # 这里的 vocab_size 必须大于等于你实际训练出来的分词器大小！
    # 如果你最终字典大小是 8000，这里填 8000 即可。
    vocab_size = 8000  
    block_size = 256   # 上下文窗口大小 (与 DataLoader 中的 block_size 保持一致)
    n_embd = 512       # 隐藏层维度
    n_head = 8         # 注意力头数 (512 / 8 = 64 维/头)
    n_layer = 4        # Transformer 块的层数

# --- 2. 因果自注意力机制 (Causal Self-Attention) ---
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # 注册下三角掩码矩阵，防止模型“偷看”未来的字
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # B: Batch, T: Time(Sequence), C: Channels(Embedding)
        
        # 算出 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 调整维度以支持多头注意力运算
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Q 乘以 K 的转置，除以根号 d 进行缩放
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 戴上“面具” (Causal Mask)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # 注意力权重乘以 V
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # 拼接多头输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

# --- 3. 前馈神经网络 (Feed Forward) ---
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# --- 4. 核心积木：Transformer Block ---
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # 先 LayerNorm 再进入模块，残差连接在外侧 (Pre-LN 架构，现代 LLM 标配)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# --- 5. 最终的 GPT 模型本体 ---
class MiniChineseGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 模型字典容器
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # 词嵌入
            wpe = nn.Embedding(config.block_size, config.n_embd), # 位置嵌入
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 多层 Transformer
            ln_f = nn.LayerNorm(config.n_embd), # 最后的层归一化
        ))
        
        # 语言模型输出头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重共享 (Weight Tying)：让输入词表和输出词表共用同一套权重，省下大量参数
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化模型参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        
        # 生成位置索引 (0, 1, 2, ..., T-1)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        
        # 提取特征并相加
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        x = tok_emb + pos_emb
        
        # 逐层穿过 Transformer
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        # 预测下一个词的概率分布
        logits = self.lm_head(x)
        
        loss = None
        # 如果传入了目标答案 (Y)，则计算交叉熵损失
        if targets is not None:
            # logits 铺平：(B*T, vocab_size)
            # targets 铺平：(B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss