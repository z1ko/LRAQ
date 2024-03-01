import torch
import torch.nn as nn
import einops as ein

# Spatial Gating Unit
class SGU(nn.Module):
    def __init__(
        self,
        channels,
        elements,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(channels)
        self.channel = nn.Linear(channels, channels * 2, bias=False)
        self.spatial = nn.Linear(elements, elements)
        nn.init.constant_(self.spatial.bias, 1.0)

    def forward(self, x):
        # (B, T, N, D).T x (N, N)

        x = self.channel(x)
        x, s = torch.split(x, split_size_or_sections=x.shape[-1] // 2, dim=-1)

        # Spatial projection
        s = self.norm(s)
        s = s.transpose(-1, -2)
        s = self.spatial(s)
        s = s.transpose(-1, -2)

        return x * s
    
class SGULayer(nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim,
        elements,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(model_dim)
        self.proj_initial = nn.Linear(model_dim, inner_dim, bias=False)
        self.proj_final = nn.Linear(inner_dim, model_dim, bias=False)
        self.activation = nn.GELU()
        self.spatial = SGU(inner_dim, elements)

    def forward(self, x): # (..., N, D)
        r = x
        x = self.norm(x)
        x = self.proj_initial(x)
        x = self.activation(x)
        x = self.spatial(x)
        x = self.proj_final(x)
        return x + r