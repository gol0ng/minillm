MiniLLM: 从零构建 20M 中文大语言模型 🚀
本项目是一个基于 PyTorch 从零手写的微型中文大语言模型（约 16M-20M 参数）。项目不仅包含了底层的 BPE 分词器训练和 GPT 架构搭建，还完整实现了预训练 (Pre-training) 与 指令微调 (Supervised Fine-Tuning, SFT) 的全流程闭环。

这是理解现代大语言模型（如 ChatGPT、LLaMA）底层原理、数据流转以及对齐训练的最佳硬核实战项目。

📁 目录结构
Plaintext
MINILLM/
├── .venv/                      # Python 虚拟环境
├── .gitignore                  # Git 忽略文件配置
├── BPEtokenizer.py             # 纯手工实现的 Byte-Level BPE 分词器训练脚本
├── dataloader.py               # PyTorch 数据加载器，负责文本切块与动态 Padding
├── generate.py                 # 模型推理脚本，支持 System Prompt、Temperature 与 Top-P 核采样
├── model.py                    # 核心大脑：包含自注意力机制、Transformer 块的 GPT 模型架构
├── my_tokenizer_merges.json    # 训练好的 BPE 词汇表合并规则 (字典)
├── prepare_data.py             # 数据准备脚本 (用于下载维基百科等无监督语料)
├── pretrain.py                 # 第一阶段：无监督预训练主循环 (让模型学会说中文)
├── sft.py                      # 第二阶段：监督指令微调主循环 (让模型学会听指令并加入 Loss Masking)
└── README.md                   # 项目说明文档
✨ 核心特性
全链路训练闭环：完整实现了 语料获取 -> 分词训练 -> 预训练 (PT) -> 指令微调 (SFT) -> 采样推理 的大模型工业级标准流程。

纯手工 BPE 分词器：不依赖现成词表，从底层字节开始统计词频并合并，生成完全适配项目语料的定制字典。

原生 GPT 架构：采用标准的 Decoder-only Transformer 架构，包含因果自注意力掩码 (Causal Mask) 和 Pre-LayerNorm 设计，并使用了 Weight Tying 共享权重技术。

指令微调与角色扮演 (System Prompt)：在 sft.py 中实现了严谨的 Loss Masking（损失屏蔽） 机制，仅对 Assistant 的回答计算梯度，支持设定 System Prompt 进行角色扮演。

先进的解码策略：推理阶段实现了 Temperature 缩放与 Top-P (Nucleus Sampling) 截断，告别无脑复读，保证生成文本的多样性与逻辑性。

🧠 模型超参数 (约 20M 参数级)
词表大小 (Vocab Size): 8000

上下文窗口 (Block Size): 256 / 512

隐藏层维度 (n_embd): 512

注意力头数 (n_head): 8

Transformer 层数 (n_layer): 4

🚀 快速开始
请按照以下顺序执行脚本，完成属于你自己的大模型炼丹之旅：

1. 环境准备
确保已激活虚拟环境，并安装必要的依赖：

Bash
pip install torch datasets
2. 第一阶段：预训练 (Pre-training)
目标：让模型阅读大量文本，学会汉字规律与基础常识。

获取语料：运行数据准备脚本，拉取预训练中文语料（例如 corpus.txt）。
或在我的modelscope里面下载：
Bash
python prepare_data.py
训练分词字典：让分词器学习语料，生成 my_tokenizer_merges.json。

Bash
python BPEtokenizer.py
启动预训练：开始自回归训练。模型权重将随着 Epoch 保存为 .pt 文件。

Bash
python pretrain.py
3. 第二阶段：监督微调 (SFT)
目标：给模型上“礼仪课”，让它学会以对话问答的形式与人类交互。

准备问答数据：在项目根目录创建一个 sft_data.jsonl，录入带有 System, User 和 Assistant 的 JSON 格式问答对。

启动微调：加载预训练权重，开启含有 Loss Masking 机制的微调训练。

Bash
python sft.py
4. 终极测试：与模型对话
加载 SFT 训练后的模型权重，输入系统设定与你的问题，见证智能的诞生。

Bash
python generate.py
📝 进阶指南与 TODO
[ ] 接入 TensorBoard 或 Weights & Biases 记录 Loss 曲线。

[ ] 收集更高质量的混合语料（百科 + 小说 + 代码）以提升模型智商。

[ ] 将模型架构升级为 LLaMA 标准（替换为 RoPE 旋转位置编码和 RMSNorm）。

Created with Passion & PyTorch.