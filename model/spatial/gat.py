import torch
import torch.nn as nn
import einops as ein

# Graph Attention Network layer
class GAT(nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim
    ):
        super().__init__()

        self.keys = nn.Linear(model_dim, inner_dim, bias=False)
        self.queries = nn.Linear(model_dim, inner_dim, bias=False)
        self.values = nn.Linear(model_dim, inner_dim, bias=False)
        self.activation = nn.GELU()

    def attention(self, x):
        K = self.keys(x)
        Q = self.queries(x)
        S = torch.matmul(Q, K.transpose(-2, -1))
        A = nn.functional.softmax(S, dim=-1)
        return A

    def forward(self, x):
        r = x
        A = self.attention(x)
        x = A @ self.values(x)
        x = self.activation(x)
        return x + r