"""\
Implementation of Linear Recurrent Unit (LRU) from:
    https://arxiv.org/pdf/2303.06349.pdf
"""

import torch
import torch.nn as nn
import einops as ein
import math

# NOTE: Should use custom kernels
from model.utils.acceleration import associative_scan, binary_operator_diag

class LRU(nn.Module):
    """ Implementation of a Linear Recurrent Unit (LRU)
        https://arxiv.org/pdf/2303.06349.pdf
    """

    def __init__(
        self,
        state_dim,                  # The state dimension is the same as the input dimension and output dimension
        r_min=0.4,                  # Min. radius in the complex plane
        r_max=0.9,                  # Max. radius in the complex plane
        phase_max=math.pi * 2,      # Phase in the form of [0, phase_max]
        **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        self.state = torch.complex(torch.zeros(state_dim), torch.zeros(state_dim))

        # Input to output, skip connection, implemented in the block
        # self.D = nn.Parameter(torch.randn([state_dim, state_dim]) / math.sqrt(state_dim))

        # Diagonal state matrix parameters
        u1 = torch.rand(state_dim)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max + r_min) * (r_max - r_min) + r_min**2)))
        u2 = torch.rand(state_dim)
        self.theta_log = nn.Parameter(torch.log(phase_max * u2))

        # Diagonal state matrix and normalization factor
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))))

        # Input to state matrix
        B_re = torch.randn([state_dim, state_dim]) / math.sqrt(2 * state_dim)
        B_im = torch.randn([state_dim, state_dim]) / math.sqrt(2 * state_dim)
        self.B = nn.Parameter(torch.complex(B_re, B_im))

        # State to output matrix
        C_re = torch.randn([state_dim, state_dim]) / math.sqrt(state_dim)
        C_im = torch.randn([state_dim, state_dim]) / math.sqrt(state_dim)
        self.C = nn.Parameter(torch.complex(C_re, C_im))

    

    def forward(self, x):  # (B, L, F)
        self.state = self.state.to(self.B.device)

        # Istantiate diagonal state matrix
        L_mod = torch.exp(-torch.exp(self.nu_log))
        L_re = L_mod * torch.cos(torch.exp(self.theta_log))
        L_im = L_mod * torch.sin(torch.exp(self.theta_log))
        L_diag = torch.complex(L_re, L_im).to(self.B.device)

        # Istantiate normalization factor
        G_norm = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        B_norm = self.B * G_norm

        L_elems = L_diag.tile(x.shape[1], 1)
        B_elems = x.to(B_norm.dtype) @ B_norm.T

        def inner_state_fn(B_seq):
            return associative_scan(binary_operator_diag, (L_elems, B_seq))[1]

        inner_states = torch.vmap(inner_state_fn)(B_elems)
        return (inner_states @ self.C.T).real

    def forward_with_state(self, x, state):  # (J, F), (J, S) dove F = S

        # TODO: (performance) cache instantiation of matrices
        L_mod = torch.exp(-torch.exp(self.nu_log))
        L_re = L_mod * torch.cos(torch.exp(self.theta_log))
        L_im = L_mod * torch.sin(torch.exp(self.theta_log))
        L_diag = torch.complex(L_re, L_im).to(self.B.device)

        G_norm = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        B_norm = self.B * G_norm

        y = torch.zeros_like(x)
        
        # state = L_diag * state + B_norm @ x.to(dtype=self.B.dtype)
        # y = (self.C @ state).real
        
        for i in range(x.shape[0]):
            state[i] = L_diag * state[i] + B_norm @ x[i].to(dtype=self.B.dtype)
            y[i] = (self.C @ state[i]).real

        return y, state

    @staticmethod
    def add_parser_args(parser):
        opts = parser.add_argument_group('linear-recurrent-unit')
        opts.add_argument('--lru_phase', type=float, default=2 * math.pi)
        opts.add_argument('--lru_radius_min', type=float, default=0.4)
        opts.add_argument('--lru_radius_max', type=float, default=0.8)



class LRULayer(nn.Module):
    """ Wrapper for an LRU. Adds skip connection, normalization and stacking.
    """

    def __init__(
        self,
        state_dim,
        dropout,
        **kwargs
    ):
        super().__init__()

        # NOTE: Uses typical Norm-RNN-Activation-Dropout layout
        self.norm = nn.LayerNorm(state_dim)
        self.lru = LRU(state_dim, **kwargs)
        self.act = nn.Sequential(
            nn.Linear(state_dim, 2 * state_dim),    # expands to use gating mechanism
            nn.GLU()
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # (B, L, F)
        residual = x

        x = self.norm(x)
        x = self.lru(x)
        x = self.act(x)
        x = self.dropout(x)

        y = x + residual  # (B, L, F)
        return y

    def forward_with_state(self, x, state):  # (N, L, F), (N, S)
        residual = x

        x = self.norm(x)
        x, state = self.lru.forward_with_state(x, state)
        x = self.act(x)
        x = self.dropout(x)

        y = x + residual
        return y, state

# Complete temporal model
class LRUModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        state_dim,
        layers,
        aggregator,
        dropout
    ):
        super().__init__()

        if aggregator == 'last':
            self.aggregator = lambda x: x[:, -1, :]
        elif aggregator == 'mean':
            self.aggregator = lambda x: torch.mean(x, dim=1).squeeze()
        elif aggregator == 'max':
            def max_pooling(x):
                x = ein.rearrange(x, 'B T D -> B D T')
                x = nn.functional.max_pool1d(x, kernel_size=99)
                x.squeeze_()
                return x
            self.aggregator = max_pooling
        else:
            raise ValueError()

        self.encoder = nn.Linear(input_dim, state_dim)
        self.decoder = nn.Linear(state_dim, output_dim)
        self.layers = nn.ModuleList([
            LRULayer(state_dim, dropout)
            for _ in range(layers)
        ])

    def forward(self, x):   # (B, T, D)        
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.aggregator(x)
        return self.decoder(x)