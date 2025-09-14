"""
简单自注意力机制
"""
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
# 第一步：计算查询向量（这里是journey）与每个输入令牌之间的点积，即计算输入查询x2相关的注意力分数
query = inputs[1]  # 2nd input token is the query
"""
inputs.shape[0] 获取 inputs 张量的第一个维度的大小，也就是6，表示有6个令牌。
torch.empty(inputs.shape[0]) 创建一个未初始化的张量，其形状为 [6] ，即一个包含6个元素的一维张量。
这个张量将用于存储查询向量（query）与每个输入令牌之间的注意力分数。
"""
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)
    print(i, attn_scores_2[i])

print(attn_scores_2)

# res = 0.
# for idx, element in enumerate(inputs[0]):
#     res += inputs[0][idx] * query[idx]

# print(res)
# print(torch.dot(inputs[0], query))

#第二步：对得到的注意力分数进行归一化，得到注意力权重
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

#使用softmax 进行归一化
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

#以上softmax_naive 在大输入值或小输入值时可能会遇到数值稳定性问题，比如溢出或者下溢，因此使用softmax的pytorch 实现
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

#第三步：使用注意力权重对输入令牌进行加权求和，得到输出向量
#将嵌入的输入词元xi 与相应的注意力权重相乘，再将得到的向量求和来计算上下文向量z2,
#上下文向量z2 是所有输入向量的加权总和，通过每个输入向量与对应的注意力权重相乘获得
query = inputs[1] # 2nd input token is the query
context_vec_2 = torch.zeros(query.shape)    #query.shape 获取查询向量的形状，即 [3]，torch.zeros(query.shape) 创建一个与查询向量形状相同的全零张量，即一个包含3个零的一维张量 [0.0, 0.0, 0.0]
for i,x_i in enumerate(inputs):
    weighted_x = attn_weights_2[i]*x_i
    print(f"步骤 {i+1}:")
    print(f"  输入向量 {i+1} (x_{i+1}): {x_i}")
    print(f"  注意力权重 {i+1}: {attn_weights_2[i]}")
    print(f"  加权后的向量 {i+1}: {weighted_x}")
    context_vec_2 += weighted_x
    print(f"  更新后的上下文向量: {context_vec_2}")
    print()
print("最终上下文向量:", context_vec_2)
#以上计算输入了2 的注意力权重和上下文向量

#计算所有词元的注意力权重和上下文向量
#1、分数
attn_scores = torch.empty((6,6))
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i, x_j)
print(attn_scores)

#for 循环较慢，使用矩阵乘法获取相同结果
attn_scores = inputs @ inputs.T
print(attn_scores)


#2、归一化
attn_weights = torch.softmax(attn_scores, dim=1)   #dim 参数指定了softmax操作应用的维度，dim=0 表示对每列进行操作（跨行）dim=1 表示对每行进行操作（跨列）
# 3.在注意力机制中，通常对注意力分数矩阵的行（ dim=1 ）做softmax，使得每行的和为1，表示不同位置对当前查询的注意力权重分布
print(attn_weights)

#3、上下文向量
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

