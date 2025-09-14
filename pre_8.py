import torch.nn as nn
import torch
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))  

    def forward(self, x):
        keys = x @ self.W_key
        values = x @ self.W_value
        query = x @ self.W_query
        attn_scores = query @ keys.T  #将查询向量和健向量相乘，得到注意力分数
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
'''
可以通过使用nn.Linear层来实现，当偏置单元被禁用时，提供了优化的权重初始化方案
'''
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out,qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  

    def forward(self, x):
        keys = self.W_key(x)
        values = self.W_value(x)
        query = self.W_query(x)
        attn_scores = query @ keys.T  #将查询向量和健向量相乘，得到注意力分数
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
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
    torch.manual_seed(123)
    d_in = 3
    d_out = 2
    sa_v1 = SelfAttention_v1(d_in, d_out)
    x = inputs
    print(sa_v1(x))

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(x))
#SelfAttention_v1 和 SelfAttention_v2 输出的结果是不同的，这是因为它们使用了不同的权重初始化方案
#SelfAttention_v1 手动初始化了查询、键、值矩阵，而 SelfAttention_v2 使用了 nn.Linear 层，默认启用了偏置项
#为了确保输出结果一致，需要在 SelfAttention_v2 中禁用偏置项