import torch
import torch.nn as nn
import einops as ein
import numpy as np
import lightning

from tqdm import tqdm

from model.spatial.gmlp import SGULayer
from model.temporal.lru import LRULayer
from model.utils.criterion import CEplusMSE, MeanOverFramesAccuracy, F1Score, EditDistance
from model.utils.visualize import visualize
from model.utils import calculate_metrics

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
        classes,
        **kwargs
    ):
        super().__init__()
        self.classes = classes

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

	    # Near temporal aggregator
        #self.conv_near = nn.Conv1d(model_dim, model_dim, kernel_size=5, padding='same')
        #self.conv_medium = nn.Conv1d(model_dim, model_dim, kernel_size=15, padding='same')
        #self.conv_far = nn.Conv1d(model_dim, model_dim, kernel_size=35, padding='same')
        #self.conv_agg = nn.Linear(model_dim * 4, model_dim)

        # Temporal evolution
        self.temporal_layers = nn.ModuleList()
        for _ in range(temporal_layers_count):
            self.temporal_layers.append(
                LRULayer(model_dim, temporal_dim, 0, False, **kwargs)
            )

        # Final frame classifier
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, self.classes),
        )

    def forward(self, x): # B T J D

        # Spatial reasoning
        x = self.embed(x)
        for layer in self.spatial_layers: 
            x = layer(x)

        x = ein.rearrange(x, 'B T J D -> B T (J D)')
        x = self.spatial_agg(x) # B T M
        
        # Local temporal kernels
        #x = ein.rearrange(x, 'B T M -> B M T')
        #a = self.conv_near(x)
        #b = self.conv_medium(x)
        #c = self.conv_far(x)
        #x = torch.cat([x, a, b, c], dim=1)

        # Condense local kernels
        #x = ein.rearrange(x, 'B M T -> B T M')
        #x = self.conv_agg(x)

        # Long range temporal reasoning
        B, T, M = x.shape
        for layer in self.temporal_layers:
            x = layer(x) # B T M
        
        # Classifier of each frame
        x = self.classifier(x)
        return x[None, ...]

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
        classes,
        **kwargs
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step = scheduler_step
        self.classes = classes

        # Criterions
        self.ce_plus_mse = CEplusMSE(num_classes=self.classes, alpha=0.17)
        self.edit = EditDistance(normalize=True)
        self.mof = MeanOverFramesAccuracy()
        self.f1 = F1Score()

        self.model = LRTAS(
            model_dim=model_dim, 
            joint_count=joint_count, 
            joint_features=joint_features,
            temporal_layers_count=temporal_layers,
            temporal_dim=temporal_state_dim,
            spatial_layers_count=spatial_layers,
            spatial_dim=spatial_state_dim,
            classes=self.classes,
            **kwargs
        )

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        from lightning.pytorch.utilities import grad_norm
        norms = grad_norm(self.model, norm_type=2)
        self.log('norm_total', norms['grad_2.0_norm_total'], prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):     # (B L S F)
        samples, targets = batch

        logits = self.model(samples) # N B T C logits
        logits = ein.rearrange(logits, "N B T C -> N B C T")
        loss = self.ce_plus_mse(logits, targets)

        self.log('train/loss', loss["loss"], prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss['loss']

    def validation_step(self, batch, batch_idx):
        samples, targets = batch

        # Get network predictions
        logits = self.model(samples)
        probs = torch.softmax(logits, dim=-1)

        predictions = torch.argmax(probs, dim=-1)
        classes = torch.max(predictions)
        assert(classes < self.classes)

        metrics = calculate_metrics(predictions[-1], targets, prefix='val')
        logits = ein.rearrange(logits, "N B T C -> N B C T")
        loss = self.ce_plus_mse(logits, targets)

        self.log('val/loss', loss['loss'], prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(metrics, prog_bar=False, on_step=False, on_epoch=True)
        return loss['loss']

    def forward(self, x):
        return self.model(x)

    def predict(self, dataset):

        scores = []
        for sample, targets in tqdm(dataset, total=len(dataset)):

            # Get network predictions
            logits = self.model(sample[None, ...])
            probs = torch.softmax(logits, dim=2)
            results = torch.argmax(probs, dim=2)

            loss_mof = self.mof(results, targets[None, :, 1])
            loss_f1 = self.f1(results, targets[None, :, 1])
            loss_edit = self.edit(results, targets[None, :, 1])

            t = np.array(targets[..., 1].cpu())
            r = np.array(results[0].cpu())
            scores.append({
                'results': r,
                'targets': t,
                'mof': loss_mof,
                'f1': loss_f1,
                'edit': loss_edit
            })

        return scores
        

    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.SGD(params=params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.scheduler_step, gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}