import torch
import json
from torch.utils.data import Dataset, DataLoader

# --- 1. 组装你的分词器大脑 ---
class MyTokenizer:
    def __init__(self, merges_path):
        # 唤醒你刚刚炼好的词汇表
        with open(merges_path, "r", encoding="utf-8") as f:
            str_merges = json.load(f)
            # 把 JSON 里存的字符串键 "228,189" 还原成元组 (228, 189)
            self.merges = {tuple(map(int, k.split(","))): v for k, v in str_merges.items()}

        # 构建反向映射：token_id -> 字节对 (用于解码)
        self.id_to_pair = {v: k for k, v in self.merges.items()}

    def encode(self, text):
        # 1. 先把文本打碎成最基础的 0-255 字节流
        tokens = list(text.encode("utf-8"))
        
        # 2. 疯狂查字典，能合并就合并
        while len(tokens) >= 2:
            # 找出当前序列里所有相邻的对
            stats = {(tokens[i], tokens[i+1]): i for i in range(len(tokens)-1)}
            
            # 在我们保存的 merges 规则里，找到优先级最高（最早学习到）的那个对
            # 如果没找到，返回无穷大
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            # 如果找出的这个对不在我们的字典里，说明已经无法再压缩了，退出循环
            if pair not in self.merges:
                break
                
            # 执行合并替换
            idx = self.merges[pair]
            tokens = self._merge_tokens(tokens, pair, idx)
            
        return tokens

    def _merge_tokens(self, ids, pair, idx):
        # 底层替换逻辑，和我们之前写的一样
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def decode(self, ids):
        """将 token id 序列解码回文本"""
        # 1. 先把所有合并后的 token 拆分回原始字节
        bytes_list = []
        for token_id in ids:
            if token_id in self.id_to_pair:
                # 这是一个合并后的 token，拆分回两个字节
                pair = self.id_to_pair[token_id]
                bytes_list.extend(pair)
            else:
                # 这是原始字节 (0-255)，直接保留
                bytes_list.append(token_id)

        # 2. 把字节流拼成 bytes，然后用 utf-8 解码成字符串
        try:
            text = bytes(bytes_list).decode("utf-8", errors="replace")
        except Exception:
            text = bytes(bytes_list).decode("utf-8", errors="replace")

        return text


# --- 2. 组装 PyTorch 数据集 ---
class WikiDataset(Dataset):
    def __init__(self, txt_path, tokenizer, block_size, max_chars=None):
        self.block_size = block_size

        # 读取全部文本（如果 max_chars 为 None，则读取全部）
        if max_chars is None:
            print(f"正在读取 {txt_path} 的全部内容...")
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            # 为了防止纯 Python 编码太慢或内存爆炸，我们只读前 max_chars 个字符来做测试
            print(f"正在读取 {txt_path} 的前 {max_chars} 个字符...")
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = f.read(max_chars)

        print("正在拼命分词压缩中... (纯Python跑得慢，请耐心等待几秒)")
        self.data = tokenizer.encode(raw_text)
        print(f"分词完毕！共生成了 {len(self.data)} 个 Token。")

    def __len__(self):
        # 减去 block_size，防止切片时越界
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        # 切出一块长度为 block_size + 1 的数据
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # X 是前 block_size 个字 (输入)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # Y 是错位1个字的后 block_size 个字 (目标)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


