import torch
import torch.nn as nn
import einops as ein
import cairo
import random
import lightning

from model.spatial.gmlp import SGULayer
from model.temporal.lru import LRULayer
from model.utils.criterion import CEplusMSE, MeanOverFramesAccuracy, F1Score

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

        # Criterions
        self.ce_plus_mse = CEplusMSE()
        self.mof = MeanOverFramesAccuracy()
        self.f1 = F1Score(25)

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
        loss = self.ce_plus_mse(results, targets)

        self.log_dict({
            "train/loss-ce+mse": loss["loss"],
            "train/loss-mse": loss["loss_mse"],
            "train/loss-ce": loss["loss_ce"],
        }, prog_bar=True)

        return loss['loss']

    def validation_step(self, batch, batch_idx):
        samples, targets = batch

        # Get network predictions
        logits = self.model(samples)
        probs = torch.softmax(logits, dim=2)
        results = torch.argmax(probs, dim=2)

        loss_mof = self.mof(results, targets)
        loss_f1 = self.f1(results, targets)
        loss_edit = self.edit(results, targets)

        self.log_dict({
            'validation/loss-mof': loss_mof,
            'validation/loss-f1': loss_f1,
            'validation/loss-edit': loss_edit
        }, prog_bar=True)

        return loss_mof + loss_f1 + loss_edit

    # Show temporal segmentation of a single sample
        #    def visualize(self, x, output_file=None):
        #        samples, targets = x
        #
        #        logits = self.model(samples)
        #        probs = torch.softmax(logits, dim=2)
        #        results = torch.argmax(probs, dim=2)
        #
        #        # Create one color for each action class
        #        colors = []
        #        for _ in range(25):
        #            colors.append((random.random(), random.random(), random.random()))
        #
        #        frames = samples.shape[0]
        #        with cairo.ImageSurface(cairo.FORMAT_ARGB32, frames, 100 + 100 + 20) as surface:
        #            ctx = cairo.Context(surface)
        #
        #            for frame in range(frames):
        #                pred, target = results[frame], targets[frame]
        #
        #                # ground truth
        #                ctx.set_source_rgb(*colors[target])
        #                ctx.move_to(float(frame), 0)
        #                ctx.line_to(float(frame), 100)
        #                ctx.stroke()
        #
        #                # result prediction
        #                ctx.set_source_rgb(*colors[pred])
        #                ctx.move_to(float(frame), 120)
        #                ctx.line_to(float(frame), 220)
        #                ctx.stroke()
        #
        #            if output_file is not None:
        #                surface.write_to_png(output_file)
        #

    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.scheduler_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
