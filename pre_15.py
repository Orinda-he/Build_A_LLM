import torch
import torch.nn as nn
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

# Some sample data
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
"""
GELU 的平滑特性可以在训练过程中带来更好的优化效果，因为它允许模型参数进行更细微
的调整。相比之下，ReLU 在零点处有一个尖锐的拐角（参见图 4-8 的右图），有时会使得优化过
程更加困难，特别是在深度或复杂的网络结构中。此外，ReLU 对负输入的输出为 0，而 GELU
对负输入会输出一个小的非零值。这意味着在训练过程中，接收到负输入的神经元仍然可以参与
学习，只是贡献程度不如正输入大。
"""