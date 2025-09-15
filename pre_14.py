"""
实现真正的层归一化，完善DummyLayerNorm
"""
import torch.nn as nn
import torch
torch.manual_seed(123)

# create 2 training examples with 5 dimensions (features) each
batch_example = torch.randn(2, 5) 

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
# print(out)
#1、计算均值和方差
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
#dim =1 或者 dim =-1 指计算列维度的平均值，以获得每行的一个平均值
#dim =0 指计算行维度的平均值，以获得每列的一个平均值
# print("Mean:\n", mean)
# print("Variance:\n", var)

#2、归一化，减去均值，并将结果除以方差的平方根
out_norm = (out - mean) / torch.sqrt(var)
# print("Normalized layer outputs:\n", out_norm)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)
#通过将 sci_mode 设置为 False 来关闭科学记数法，从而在打印张量值时避免使用科学记数法
torch.set_printoptions(sci_mode=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)

#层归一化类
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# if __name__ == "__main__":
#     ln = LayerNorm(emb_dim=5)
#     out_ln = ln(batch_example)
#     mean = out_ln.mean(dim=-1, keepdim=True)
#     var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

#     print("Mean:\n", mean)
#     print("Variance:\n", var)
