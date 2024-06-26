#import matplotlib.pyplot as plt
#import wandb
import numpy as np

from lightning.pytorch import Trainer
#from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

#from model.utils.args import base_arg_parser
from dataset.assembly import AssemblyTSMModule
from model.utils.visualize import visualize
from model.tas import TSMTAS

#parser = base_arg_parser()
#opts = parser.parse_args()
#print(vars(opts))

#d = AssemblyDataModule(
#    'data/processed/assembly', 
#    batch_size=6,
#    window_size=2000
#)
#d.setup()
#
#model = TemporalActionSegmentation(
#    model_dim=100,
#    learning_rate=0.00001,
#    weight_decay=0.0005,
#    scheduler_step=30,
#    joint_count=2*21,
#    joint_features=3,
#    temporal_state_dim=256,
#    spatial_state_dim=64,
#    temporal_layers=6,
#    spatial_layers=2
#)

#logger = None
#if opts.dashboard:
#    logger = WandbLogger(save_dir='runs/')
#    logger.log_hyperparams(opts)

d = AssemblyTSMModule(
    path_to_data='/media/z1ko/2TM2/datasets/Assembly101',
    views=['C10095_rgb'],
    batch_size=8,
)
d.setup()

model = TSMTAS(
    learning_rate=0.01,
    scheduler_step=150
)

logger = WandbLogger(save_dir='runs/', name='TSMTAS')
lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint = ModelCheckpoint(
    filename='checkpoints',
    save_top_k=1,
    verbose=True,
    monitor='val/mof',
    mode='max'
)

trainer = Trainer(
    logger=logger,
    max_epochs=600,
    callbacks=[lr_monitor, checkpoint]
)

# Qualitative results
#trainer.predict(model, d.val_dataloader())

# Train
trainer.fit(
    model,
    train_dataloaders=d.train_dataloader(),
    val_dataloaders=d.val_dataloader()
)
