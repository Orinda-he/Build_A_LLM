"""
利用因果注意力隐藏未来词汇（Causal Attention）掩码注意力
希望自注意力机制在预测序列中的下一个词元时，只考虑当前位置之前的词元
它限制模型在处理任何给定词元时，只能基于序列中先前和当前输入来计算注意力分数，而标准的自注意力机制可以一次性访问整个输入序列
"""
import torch.nn as nn
import torch
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
    torch.manual_seed(789)
    d_in = 3
    d_out = 2
    sa_v2 = SelfAttention_v2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    values = sa_v2.W_value(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    print(attn_weights)

    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length,context_length))
    mask_simple = attn_weights*mask_simple
    print(mask_simple)

    row_sums = mask_simple.sum(dim=-1, keepdim=True)
    mask_simple_norm = mask_simple / row_sums
    print(mask_simple_norm)
    
#在因果注意力中，获得掩码后的注意力权重矩阵的一种更有效的方法是在应用softmax函数之前，将注意力分数矩阵中的上三角部分设置为负无穷大
    mask = torch.triu(torch.ones(context_length,context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(),-torch.inf)
    print(masked)
    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
    print(attn_weights)

    torch.manual_seed(123)
    dropout = nn.Dropout(0.5)
    attn_weights = dropout(attn_weights)
    #为了补偿减少的活跃元素，矩阵中剩余的值会按1/0.5 的比例进行放大，这种放大对于维持注意力权重的整体平衡非常重要，可以确保在
    # 训练和推理过程中，模型的注意力机制的平均影响保持一致
    print(attn_weights)

# 利用dropout掩码额外的注意力权重，有效减少过拟合，dropout仅在训练过程中使用
#两个特定的时间点，分别是在计算注意力权重之后，将这些权重应用于值向量之后
    # torch.manual_seed(123)
    # dropout = nn.Dropout(0.5)
    # example = torch.ones(6,6) #创建一个全1矩阵
    # print(dropout(example))
