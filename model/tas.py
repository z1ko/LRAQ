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

	    # Near temporal aggregator
        #self.conv_near = nn.Conv1d(model_dim, model_dim, kernel_size=5, padding='same')
        #self.conv_medium = nn.Conv1d(model_dim, model_dim, kernel_size=15, padding='same')
        #self.conv_far = nn.Conv1d(model_dim, model_dim, kernel_size=35, padding='same')
        #self.conv_agg = nn.Linear(model_dim * 4, model_dim)

        # Temporal evolution
        self.temporal_layers = nn.ModuleList()
        for _ in range(temporal_layers_count):
            self.temporal_layers.append(
                LRULayer(model_dim, temporal_dim, 30, True, **kwargs)
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
        return x

class TSMLRTAS(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.model_dim = 64
        self.stages = 5

        self.norm = nn.LayerNorm(2048)
        self.embed = nn.Linear(2048, self.model_dim)

        self.temporal_layers = nn.ModuleList()
        for _ in range(self.stages):
            self.temporal_layers.append(
                LRULayer(self.model_dim, 128, 15, True, **kwargs)
            )

        self.classifiers = nn.ModuleList()
        for _ in range(self.stages):
            self.classifiers.append(
                nn.Linear(self.model_dim, 202)
            )

    def forward(self, x): # [B, T, D]
        B, T, D = x.shape

        x = self.norm(x)
        x = self.embed(x)
        
        stages = torch.zeros((len(self.temporal_layers), B, T, 202)).cuda()
        for s, layer in enumerate(self.temporal_layers):
            x = layer(x)
            stages[s] = self.classifiers[s](x) # Save classification of stage

        return stages

class TSMTAS(lightning.LightningModule):
    def __init__(self, learning_rate, **kwargs):
        super().__init__()

        self.learning_rate = learning_rate
        #self.weight_decay = weight_decay
        #self.scheduler_step = scheduler_step
        self.counter = 0

        # Criterions
        self.ce_plus_mse = CEplusMSE(num_classes=202, alpha=0.17)
        self.edit = EditDistance(normalize=True)
        self.mof = MeanOverFramesAccuracy()
        self.f1 = F1Score(num_classes=202)

        self.model = TSMLRTAS(**kwargs)

    def metrics(self, stage, logits, targets):

        probs = torch.softmax(logits, dim=2)
        res = torch.argmax(probs, dim=2)

        loss_mof = self.mof(res, targets)
        loss_f1 = self.f1(res, targets)
        loss_edit = self.edit(res, targets)

        return {
            f'{stage}/F1@10': loss_f1['F1@10'],
            f'{stage}/F1@25': loss_f1['F1@25'],
            f'{stage}/F1@50': loss_f1['F1@50'],
            f'{stage}/edit': loss_edit,
            f'{stage}/mof': loss_mof,
        }

    def training_step(self, batch, batch_idx):     # (B L S F)
        samples, targets, _ = batch

        logits = self.model(samples) # B T C logits
        metrics = self.metrics('train', logits[-1], targets)
        logits = ein.rearrange(logits, "N B T C -> N B C T")
        loss = self.ce_plus_mse(logits, targets)

        self.log_dict(metrics, prog_bar=True, batch_size=samples.shape[0])
        self.log('train/loss', loss['loss'], on_step=True, on_epoch=True, logger=True)

        return loss['loss']
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        from lightning.pytorch.utilities import grad_norm
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def validation_step(self, batch, batch_idx):
        samples, targets, metadata = batch

        # Get network predictions
        logits = self.model(samples)
        metrics = self.metrics('val', logits[-1], targets)
        self.log_dict(metrics, prog_bar=True, batch_size=samples.shape[0])
        
        return metrics
    
    def predict_step(self, batch, batch_idx):
        samples, targets, metadata = batch

        logits = self.model(samples)
        probs = torch.softmax(logits, dim=2)
        results = torch.argmax(probs, dim=2)

        # Save visualization
        for i in range(results.shape[0]):
            m = metadata[i]
            t = targets[i]
            r = results[i]

            loss_mof = self.mof(results[None, :], targets[None, :])
            loss_f1 = self.f1(results[None, :], targets[None, :])
            loss_edit = self.edit(results[None, :], targets[None, :])

            savepath = f"data/output/mof_{loss_mof}-f1@10_{loss_f1['F1@10']}-edit_{loss_edit}-{self.counter:010d}.png"
            self.counter += 1

            t = np.array(t.cpu())
            r = np.array(r.cpu())
            visualize(r, t, 202, savepath)


    def configure_optimizers(self):
        params = list(self.model.parameters())
        #optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        #optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(params=params, lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.scheduler_step, gamma=0.1)
        return {'optimizer': optimizer } #'lr_scheduler': scheduler}


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

        # Criterions
        self.ce_plus_mse = CEplusMSE(num_classes=25, alpha=0.17)
        self.edit = EditDistance(normalize=True)
        self.mof = MeanOverFramesAccuracy()
        self.f1 = F1Score(num_classes=25)

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

    def training_step(self, batch, batch_idx):     # (B L S F)
        samples, targets = batch

        results = self.model(samples) # B T C logits
        results = ein.rearrange(results, "B T C -> B C T")
        loss = self.ce_plus_mse(results, targets[:, :, 1])

        self.log('train/loss-ce+mse', loss["loss"], prog_bar=True)
        self.log_dict({
            "train/loss-mse": loss["loss_mse"],
            "train/loss-ce": loss["loss_ce"],
        }, prog_bar=False)

        return loss['loss']

    def validation_step(self, batch, batch_idx):
        samples, targets = batch

        # Get network predictions
        logits = self.model(samples)
        probs = torch.softmax(logits, dim=2)
        results = torch.argmax(probs, dim=2)

        loss_mof = self.mof(results, targets[:, :, 1])
        loss_f1 = self.f1(results, targets[:, :, 1])
        loss_f1_total = loss_f1["F1@10"] + loss_f1["F1@25"] + loss_f1["F1@50"]
        loss_edit = self.edit(results, targets[:, :, 1])
        loss_total = loss_mof + loss_f1_total + loss_edit

        self.log('validation/loss-total', loss_total, prog_bar=True)
        self.log_dict({
            'validation/loss-mof': loss_mof,
            'validation/loss-edit': loss_edit,
            'validation/loss-F1@10': loss_f1['F1@10'],
            'validation/loss-F1@25': loss_f1['F1@25'],
            'validation/loss-F1@50': loss_f1['F1@50'],
            'validation/loss-F1': loss_f1_total,
        }, prog_bar=False)

        return loss_total

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
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.scheduler_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
