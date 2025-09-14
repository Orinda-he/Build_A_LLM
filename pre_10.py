"""
实现一个简化的因果注意力类

"""
import torch.nn as nn
import torch
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', 
                            torch.triu(torch.ones(context_length,context_length), 
                            diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        query = self.W_query(x)
        attn_scores = query @ keys.transpose(1,2) #将查询向量和健向量相乘，得到注意力分数
        attn_scores = attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

if __name__ == '__main__':
    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0) #为了简单起见，可以通过复制输入文本示例来模拟批量输入，batch.shape = [2,5,3]
    torch.manual_seed(123)
    d_in = 3
    d_out = 2
    context_length = batch.shape[1] # 6
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vec = ca(batch)
    print(context_vec)
    print(context_vec.shape)

