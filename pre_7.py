"""
实现带可训练权重的自注意力机制，缩放点积注意力（scaled dot-product attention）
"""
#之前的注意力机制目标：希望上下文向量计算为某个特定输入元素对于序列中所有输入向量的加权和

#缩放点积注意力 引入在模型训练期间更新的权重矩阵

#1、逐步计算注意力权重，引入3个可训练的权重矩阵 W_q, W_k, W_v，这三个矩阵用于将嵌入的输入词元xi映射到查询、键、值空间
#2、计算查询、键、值
#3、计算注意力分数
#4、归一化注意力分数
#5、计算上下文向量

import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
x_2 = inputs[1]
d_in = inputs.shape[1]    #输入嵌入维度d_in = 3
d_out = 2    #输出嵌入维度d_out = 2
torch.manual_seed(123) #设置固定随机种子，确保每次运算得到相同的query_2 的值
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) #初始化一个查询变换矩阵 W_query ，其维度为(输入维度×输出维度)，这里是3×2
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
query_2 = x_2 @ W_query   #@ 为torch中 乘法运算符
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
print(keys,"\nkeys.shape:",keys.shape)
print(values,"\nvalues.shape:",values.shape)

#计算注意力分数是一种点积运算，通过各自权重矩阵变幻后的查询向量和建向量进行计算
#点积运算的结果是一个标量，用于表示查询向量与键向量的相似度
key_2 = keys[1]
attn_scores_22 = query_2 @ key_2.T
print(attn_scores_22)

#通过矩阵乘法将这个计算推广到所有注意力分数
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

#将注意力分数转换为注意力权重，通过缩放注意力分数并应用softmax函数计算注意力权重
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

#计算上下文向量
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

