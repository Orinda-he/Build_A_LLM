import torch
import torch.nn as nn
from pre_18 import TransformerBlock
from pre_14 import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
'''
if __name__ == "__main__":
    GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小
    "context_length": 1024, # 上下文长度
    "emb_dim": 768,         # 嵌入维度
    "n_heads": 12,          # 注意力头数
    "n_layers": 12,         # 层数
    "drop_rate": 0.1,       # Dropout 率
    "qkv_bias": False       # Query-Key-Value 偏置
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    
    # 创建一个示例批次
    # batch_size = 2
    # seq_len = 4
    # batch = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (batch_size, seq_len))
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0) 
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")       #统计模型参数数量，目标参数量为1.24亿，这里输出为1.63亿

#原因在于原始 GPT-2 架构中使用了一个叫作权重共享（weight tying）的概念。也就是说，原始 GPT-2 架构是将词元嵌入层作为输出层重复使用的

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
#以上两个层的权重张量具有相同的形状
#Token embedding layer shape: torch.Size([50257, 768])
#Output layer shape: torch.Size([50257, 768])

    
#由于分词器词汇表中有 50 257 个条目，因此词元嵌入层和输出层非常庞大。根据权重共享的概
#念，我们需要从总的 GPT-2 模型参数计数中减去输出层的参数量：
3权重共享可以减少模型的总体内存占用和计算复杂度

    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")        #124,412,160    


#计算一下 GPTModel 对象中 1.63 亿个参数的内存需求：
#计算总的字节大小（假设每个参数是占用4字节的32 位浮点数）
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)

print(f"Total size of the model: {total_size_mb:.2f} MB")  #转换为兆字节（MB） 621.83 MB

"""
在不修改代码
的情况下，只需更新配置文件，即可使用 GPTModel 类实现 GPT-2 medium（具有 1024 维嵌
入、24 个 Transformer 块和 16 个多头注意力头）、GPT-2 large（具有 1280 维嵌入、36 个
Transformer 块和 20 个多头注意力头）和 GPT-2 xl（具有 1600 维嵌入、48 个 Transformer 块和
25 个多头注意力头）。同时，计算每个 GPT 模型的参数总数。
"""
'''
    





