import torch
import torch.nn as nn
from pre_15 import GELU
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# if __name__ == "__main__":
#     cfg = {
#         "emb_dim": 768,
#         "drop_rate": 0.1,
#         "qkv_bias": False,
#     }
#     ffn = FeedForward(cfg)
#     x = torch.randn(2,3, 768)
#     print(ffn(x).shape)           #可以看到输出张量形状与输入张量形状保持一致torch.Size([2, 3, 768])

