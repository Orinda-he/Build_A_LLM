"""
下快捷连接（也称为“跳跃连接”或“残差连接”）的概念。快捷连接最初用
于计算机视觉中的深度网络（特别是残差网络），目的是缓解梯度消失问题。梯度消失问题指的
是在训练过程中，梯度在反向传播时逐渐变小，导致早期网络层难以有效训练。

快捷连接通过跳过一个或多个层，为梯度在网络中的流动提供了一条可替代
且更短的路径。这是通过将一层的输出添加到后续层的输出中实现的。这也是为什么这种连接被
称为跳跃连接。在反向传播训练中，它们在维持梯度流动方面扮演着至关重要的角色。
"""
import torch
import torch.nn as nn
from pre_15 import GELU

#具有 5 层的深度神经网络，每层由一个线性层和一个 GELU 激活函数组成
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

#实现一个用于在模型的反向传播过程中计算梯度的函数
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

if __name__ == "__main__":
    layer_sizes = [3, 3, 3, 3, 3, 1]  

    sample_input = torch.tensor([[1., 0., -1.]])
#print_gradients 函数的输出显示，梯度在从最后一层（layers.4）到第 1 层（layers.0）的过程中逐渐变小，这种现象称为梯度消失问题
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
    )
    print_gradients(model_without_shortcut, sample_input)
    
    print("With shortcut:")
#将use_shortcut=True，模型的梯度消失问题得到了缓解，包含跳跃连接
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
    )
    print_gradients(model_with_shortcut, sample_input)