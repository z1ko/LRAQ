import torch
import torch.nn as nn
import einops as ein
import numpy as np
import lightning
import math

from tqdm import tqdm

from model.spatial.gmlp import SGULayer
from model.temporal.lru import LRULayer
from model.utils.criterion import CEplusMSE, MeanOverFramesAccuracy, F1Score, EditDistance
from model.utils.visualize import visualize
from model.utils import calculate_metrics

class TSMLRTAS(nn.Module):
    def __init__(self, dropout=0.20, **kwargs):
        super().__init__()

        self.model_dim = 64
        self.stages = 20

        self.norm = nn.LayerNorm(2048)
        self.embed = nn.Linear(2048, self.model_dim)
        self.dropout = nn.Dropout(dropout)

        self.temporal_layers = nn.ModuleList()
        for _ in range(self.stages):
            self.temporal_layers.append(
                LRULayer(self.model_dim, 128, 0, None, dropout, phase_max=math.pi/50, **kwargs)
            )

        self.classifier = nn.Linear(self.model_dim, 202)
        #self.classifiers = nn.ModuleList()
        #for _ in range(self.stages):
        #    self.classifiers.append(
        #        nn.Linear(self.model_dim, 202)
        #    )

    def forward(self, x): # [B, T, D]
        B, T, D = x.shape

        x = self.norm(x)
        x = self.dropout(x)
        x = self.embed(x)
        
        #stages = torch.zeros((len(self.temporal_layers), B, T, 202)).cuda()
        for s, layer in enumerate(self.temporal_layers):
            x = layer(x)
            #stages[s] = self.classifiers[s](x) # Save classification of stage

        #return stages
        return self.classifier(x)[None, ...]

class TSMTAS(lightning.LightningModule):
    def __init__(self, learning_rate, scheduler_step, classes, **kwargs):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = 1e-5
        self.scheduler_step = scheduler_step
        self.classes = classes
        self.counter = 0

        # Criterions
        self.ce_plus_mse = CEplusMSE(num_classes=self.classes, alpha=0.20)
        self.edit = EditDistance(normalize=True)
        self.mof = MeanOverFramesAccuracy()
        self.f1 = F1Score()

        self.model = TSMLRTAS(dropout=0.5, **kwargs)

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

        self.log_dict(metrics, prog_bar=False, on_step=False, on_epoch=True, batch_size=samples.shape[0])
        self.log('train/loss', loss['loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=samples.shape[0])
        return loss['loss']
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        #from lightning.pytorch.utilities import grad_norm
        #norms = grad_norm(self.model, norm_type=2)
        #self.log_dict(norms)
        pass

    def validation_step(self, batch, batch_idx):
        samples, targets, metadata = batch

        # Get network predictions
        logits = self.model(samples)
        metrics = self.metrics('val', logits[-1], targets)
        logits = ein.rearrange(logits, "N B T C -> N B C T")
        loss = self.ce_plus_mse(logits, targets)

        self.log_dict(metrics, prog_bar=False, on_step=False, on_epoch=True, batch_size=samples.shape[0])
        self.log('val/loss', loss['loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=samples.shape[0])
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
        optimizer = torch.optim.SGD(params=params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.scheduler_step, gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

