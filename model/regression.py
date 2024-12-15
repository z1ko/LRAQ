import torch
import torch.nn as nn
import einops as ein
import lightning

from model.temporal.lru import LRU, LRUBlock, GLU
from model.temporal.mingru import minGRU
from model.spatial.gmlp import SGULayer
from model.gtt import G2TAQ

# How to model the temporal data
#class ConvLRUBlock(nn.Module):
#    def __init__(
#        self, 
#        input_dim,      # Input dimensions
#        output_dim,     # Ouput dimensions
#        #kernel_size,    # Temporal convolutional kernel size
#        dropout,        # Dropout
#        **kwargs
#    ):
#        super().__init__()
#
#        self.conv = nn.Conv1d(input_dim, output_dim, 1)
#        self.rnn = LRULayer(
#            state_dim=output_dim,
#            dropout=dropout,
#            activation='gelu'
#        )
#        
#    def forward(self, x):   # (B L F)
#
#        # Aggregate near features
#        x = ein.rearrange(x, 'B L F -> B F L')        
#        x = self.conv(x)
#        x = ein.rearrange(x, 'B F L -> B L F')
#
#        return self.rnn(x)


class GATBlock(nn.Module):
    def __init__(
        self,
        input_dim,      # Input dimensions
        inner_dim,      # Dimension of internal projection
        output_dim,     # Output dimensions
        #heads,         # Number of heads
        dropout         # Dropout
    ):
        super().__init__()

        self.key = nn.Linear(input_dim, inner_dim, bias=False)
        self.query = nn.Linear(input_dim, inner_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)
        self.residual = nn.Linear(input_dim, output_dim)

        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def self_attention(self, x):                    # (B J F)

        K = self.key(x)                             # (B J K)
        Q = self.query(x)                           # (B J Q)
        V = self.value(x)                           # (B J V)

        S = torch.matmul(Q, K.transpose(-2, -1))    # (B J J)
        W = nn.functional.softmax(S, dim=-1)

        return torch.matmul(W, V)                   # (B J V)

    def forward(self, x):                           # (B J F)
        residual = self.residual(x)

        x = self.norm(x)
        x = self.self_attention(x)
        x = self.act(x)
        x = self.dropout(x)
        
        return x + residual
    
class LocalConv(nn.Module):
    def __init__(self, kernel_sizes, model_dim):
        super().__init__()
        
        self.mixer = nn.Conv2d(model_dim * (len(kernel_sizes) + 1), model_dim, (1,1))
        self.kernels = nn.ModuleList()
        for size in kernel_sizes:
            self.kernels.append(
                nn.Conv2d(model_dim, model_dim, (size, 1), padding='same')
            )

    def forward(self, x):
        x = ein.rearrange(x, 'B T J D -> B D T J')
        results = [ kernel(x) for kernel in self.kernels ]
        x = torch.cat([x] + results, dim=1)
        x = self.mixer(x)
        x = ein.rearrange(x, 'B D T J -> B T J D')
        return x


class LRGAModel(nn.Module):
    def __init__(
        self,
        joint_count,
        joint_features,
        model_dim,
        temporal_state_dim,
        temporal_layers,
        spatial_layers,
        dropout,
        batch_size,
        no_conv,
        temporal_method,
        spatial_method,
        **kwargs
    ):
        super().__init__()
        self.joint_count = joint_count
        self.use_conv = not no_conv
        self.temporal_method = temporal_method
        self.spatial_method = spatial_method
        self.temporal_layers = temporal_layers
        self.batch_size = batch_size
        self.model_dim = model_dim

        # Initial feature transformation
        self.embed = nn.Linear(joint_features, model_dim)

        if self.use_conv:
            self.local_convolution = LocalConv([3, 5, 9], model_dim)

        # Create stacked temporal blocks
        if temporal_method == 'LRU':
            self.temporal_model = LRUBlock(
                input_dim=model_dim,
                output_dim=model_dim,
                state_dim=temporal_state_dim,
                layers_count=temporal_layers,
                dropout=dropout,
                **kwargs
            )
        elif temporal_method == 'RNN':
            self.temporal_model = nn.RNN(
                input_size=model_dim,
                hidden_size=model_dim,
                num_layers=temporal_layers,
                dropout=dropout,
                batch_first=True,
            )
        elif temporal_method == 'GRU':
            self.temporal_model = nn.GRU(
                input_size=model_dim,
                hidden_size=model_dim,
                num_layers=temporal_layers,
                dropout=dropout,
                batch_first=True
            )
        elif temporal_method == 'LSTM':
            self.temporal_model = nn.LSTM(
                input_size=model_dim,
                hidden_size=model_dim,
                num_layers=temporal_layers,
                dropout=dropout,
                batch_first=True
            )
        elif temporal_method == 'MINGRU':
            self.temporal_model = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    minGRU(model_dim),
                    nn.ReLU(),
                )
                for _ in range(temporal_layers)
            ])
        
        # Create spaial block
        if spatial_method == 'gMLP':
            self.spatial_layers = nn.ModuleList([
                SGULayer(
                    model_dim=model_dim,
                    inner_dim=64,
                    elements=joint_count,
                    **kwargs
                ) for _ in range(spatial_layers)
            ])

        self.regressor = nn.Sequential(
            nn.Linear(model_dim, model_dim * 3),
            nn.LeakyReLU(),
            nn.Linear(model_dim * 3, 1),
            nn.Sigmoid()
        )

    def forward_temporal(self, x): # (B J) L F
        if self.temporal_method == 'LRU':
            return self.temporal_model(x)
        elif self.temporal_method == 'RNN':
            y, _ = self.temporal_model(x)
            return y[:, -1, :]
        elif self.temporal_method == 'GRU':
            y, _ = self.temporal_model(x)
            return y[:, -1, :]
        elif self.temporal_method == 'LSTM':
            y, _ = self.temporal_model(x)
            return y[:, -1, :]
        elif self.temporal_method == 'MINGRU':
            for layer in self.temporal_model:
                x = layer(x)
            return x[:, -1, :]


    def forward(self, x):                               # (B L J F)
        B, L, J, F = x.shape

        # Aggregate local features and encode features
        x = self.embed(x)

        if self.use_conv:
            x = self.local_convolution(x)

        # Temporal model, same for all joints
        x = ein.rearrange(x, 'B L J F -> (B J) L F')
        x = self.forward_temporal(x)
 
        # Spatial model
        x = ein.rearrange(x, '(B J) F -> B J F', B=B, J=J)

        if self.spatial_method == 'gMLP':
            for layer in self.spatial_layers:
                x = layer(x)

        # Aggregate in space and final output
        x = torch.mean(x, dim=1)

        #x = ein.rearrange(x, 'B J F -> B (J F)') 
        #x = ein.rearrange(x, 'B J F -> B F J')
        #x = nn.functional.max_pool1d(x, self.joint_count).squeeze()
        
        return self.regressor(x).squeeze(dim=-1)


class LRGA(lightning.LightningModule):
    def __init__(
        self,
        learning_rate,
        weight_decay,
        scheduler_step,
        maximum_quality,
        joint_count,
        joint_features,
        model_dim,
        temporal_state_dim,
        temporal_layers,
        spatial_layers,
        dropout,
        **kwargs
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step = scheduler_step
        self.maximum_quality = maximum_quality

        # Model
        self.model = LRGAModel(
            joint_count, 
            joint_features,
            model_dim, 
            temporal_state_dim,
            temporal_layers,
            spatial_layers, 
            dropout,
            **kwargs
        )

        #self.model = G2TAQ(
        #    model_dim=32, 
        #    joint_count=joint_count, 
        #    joint_feaures=joint_features,
        #    temporal_dim=32, 
        #    spatial_dim=32
        #)

        self.validation_losses = []

    def forward(self, samples):
        return self.model(samples) * self.maximum_quality

    def training_step(self, batch, batch_idx):     # (B L J F)
        samples, targets = batch

        criterion = nn.HuberLoss(reduction='mean', delta=1.5)
        results = self.forward(samples)
        loss = criterion(results, targets)

        self.log("train/loss-huber", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch

        criterion = nn.L1Loss()
        results = self.forward(samples)
        loss = criterion(results, targets)

        self.log("validation/loss-mae", loss, prog_bar=True)
        self.validation_losses.append(loss.item())
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.validation_losses.clear()
    #    epoch_validation_loss = sum(self.validation_step_outputs) / len(self.validation_step_outputs)
    #    self.best_validation_loss = min(self.best_validation_loss, epoch_validation_loss)
    #    self.log("validation/best-loss-mae", self.best_validation_loss, prog_bar=True, on_step=False, on_epoch=True)
    #    self.validation_step_outputs.clear()

    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.scheduler_step, gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def add_parser_args(parser):
        LRU.add_parser_args(parser)

        opts = parser.add_argument_group('model')
        opts.add_argument('--model_dim', type=int, default=256)
        opts.add_argument('--joint_features', type=int, default=6)
        opts.add_argument('--temporal_state_dim', type=int, default=128)
        opts.add_argument('--temporal_layers', type=int, default=4)
        opts.add_argument('--spatial_layers', type=int, default=2)
        opts.add_argument('--dropout', type=float, default=0.1)
        opts.add_argument('--temporal_method', type=str, default='LRU')
        opts.add_argument('--spatial_method', type=str, default='gMLP')
        opts.add_argument('--no_conv', action='store_true')