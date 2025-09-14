'''
将先前实现的因果注意力扩展到多个头上，每个头都有自己的查询、键和值权重矩阵
每个头的输出被连接起来，形成最终的上下文向量
多头注意力的主要思想是并行运行注意力机制，每次使用学到的不同的线性投影，这些是通过将输入数据乘以权重矩阵的到的
'''
import torch.nn as nn
import torch
from pre_10 import CausalAttention
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

if __name__ == '__main__':
    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0) #为了简单起见，可以通过复制输入文本示例来模拟批量输入，batch.shape = [2,6,3]
    torch.manual_seed(123)
    context_length = batch.shape[1] # 词元数量6
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, 2)
    context_vecs = mha(batch)
    print(context_vecs)
    '''
    结果中的第一维是2，因为有两个输入文本（文本是重复的，所以这些上下文向量完全相同）
    第二维表示每个输入中的6个词元
    第三维表示每个词元的四维嵌入
    '''
    print(context_vecs.shape)
