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
        r_min=0.8,                  # Min. radius in the complex plane
        r_max=0.99,                 # Max. radius in the complex plane
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

class CausalConv1d(nn.Module):
    """
    Causal convolution per easier online inference
    """
    
    def __init__(self, input_dim, output_dim, kernel_size, dilation=1):
        super().__init__()

        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, 
                              padding=pad, dilation=dilation)

    def forward(self, x): # [B, D, T]
        x = self.conv(x)
        x = x[:, :, :-self.conv.padding[0]] # remove trailing padding
        return 


class LRULayer(nn.Module):
    def __init__(
        self,
        model_dim,
        state_dim,
        temporal_k_size,
        use_temporal_conv,
        **kwargs
    ):
        super().__init__()
        self.use_temporal_conv = use_temporal_conv

        self.norm = nn.LayerNorm(model_dim)
        self.proj_initial = nn.Linear(model_dim, state_dim * 2, bias=False)
        self.activation = nn.GELU()
        
        self.conv = nn.Conv1d(state_dim, state_dim, kernel_size=temporal_k_size, padding='same')
        #self.conv = CausalConv1d(state_dim, state_dim, kernel_size=temporal_k_size, dilation=2)

        self.temporal = LRU(state_dim, **kwargs)
        self.proj_final = nn.Linear(state_dim, model_dim)

    def forward(self, x): # (B, T, D)
        r = x

        x = self.norm(x)
        x = self.proj_initial(x)
        x, v = torch.split(x, split_size_or_sections=x.shape[-1] // 2, dim=-1)  
        
        if self.use_temporal_conv:
            v = v.transpose(-1, -2)
            v = self.conv(v)
            v = v.transpose(-1, -2)

        v = self.temporal(v)
        x = self.activation(x)
        x = x * v

        y = self.proj_final(x)
        return y + r
