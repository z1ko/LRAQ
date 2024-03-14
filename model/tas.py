import torch
import torch.nn as nn
import einops as ein
import lightning

from model.spatial.gmlp import SGULayer
from model.temporal.lru import LRULayer
from model.utils.criterion import mean_over_frames

class LRTAS(nn.Module):
    def __init__(
        self, 
        model_dim, 
        spatial_dim, 
        spatial_layers_count,
        temporal_dim,
        temporal_layers_count,
        joint_features,
        joint_count,
        **kwargs
    ):
        super().__init__()

        self.embed = nn.Linear(joint_features, model_dim)

        # Per frame feature extractor
        self.spatial_layers = nn.ModuleList()
        for _ in range(spatial_layers_count):
            self.spatial_layers.append(
                SGULayer(model_dim, spatial_dim, joint_count, **kwargs)
            )

        # Compress spatial data
        # TODO: Maybe too big for the task
        self.spatial_agg = nn.Sequential(
            nn.Linear(model_dim * joint_count, model_dim * 5),
            nn.GELU(),
            nn.Linear(model_dim * 5, model_dim)
        )
        
        # Temporal evolution
        self.temporal_layers = nn.ModuleList()
        for _ in range(temporal_layers_count):
            self.temporal_layers.append(
                LRULayer(model_dim, temporal_dim, 5, True, **kwargs)
            )

        # Final frame classifier
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 25),
        )

    def forward(self, x): # B T J D

        # Spatial reasoning
        x = self.embed(x)
        for layer in self.spatial_layers: 
            x = layer(x)

        x = ein.rearrange(x, 'B T J D -> B T (J D)')
        x = self.spatial_agg(x) # B T M

        # Temporal reasoning
        B, T, M = x.shape
        for layer in self.temporal_layers:
            x = layer(x) # B T M
        
        # Classifier of each frame
        x = self.classifier(x)
        return x

class TemporalActionSegmentation(lightning.LightningModule):
    def __init__(
        self,
        learning_rate,
        weight_decay,
        scheduler_step,
        joint_count,
        joint_features,
        model_dim,
        temporal_state_dim,
        temporal_layers,
        spatial_state_dim,
        spatial_layers,
        **kwargs
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step = scheduler_step

        self.model = LRTAS(
            model_dim=model_dim, 
            joint_count=joint_count, 
            joint_features=joint_features,
            temporal_layers_count=temporal_layers,
            temporal_dim=temporal_state_dim,
            spatial_layers_count=spatial_layers,
            spatial_dim=spatial_state_dim,
            **kwargs
        )

    def forward(self, x):
        logits = self.model(x) # B T C
        #probs = torch.softmax(logits, dim=2)
        #prediction = torch.argmax(probs, dim=2)
        #return prediction.to(dtype=torch.float32)
        return logits

    def training_step(self, batch, batch_idx):     # (B L S F)
        samples, targets = batch

        results = self.forward(samples)

        # Frame wise cross entropy loss
        frame_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        results = ein.rearrange(results, 'B T C -> B C T')
        targets = ein.rearrange(targets, 'B T C -> B C T')
        loss = frame_criterion(results, targets[:, 1, :])

        #results = self.forward(samples)
        #loss = mean_over_frames(results, targets[:, :, 1])

        self.log("train/loss-ce", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch

        results = self.forward(samples)

        # Frame wise cross entropy loss
        frame_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        results = ein.rearrange(results, 'B T C -> B C T')
        targets = ein.rearrange(targets, 'B T C -> B C T')
        loss = frame_criterion(results, targets[:, 1, :])

        #results = self.forward(samples)
        #loss = mean_over_frames(results, targets[:, :, 1])

        self.log("validation/loss-ce", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.scheduler_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}