#import matplotlib.pyplot as plt
#import wandb
import numpy as np

from lightning.pytorch import Trainer
#from lightning.pytorch.utilities.model_summary import ModelSummary
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
    path_to_data='data/processed/assembly101',
    views=['C10095_rgb'],
    batch_size=10,
)
d.setup()

model = TSMTAS(
    learning_rate=0.001,
    scheduler_step=30
)

logger = WandbLogger(save_dir='runs/', name='TSMTAS')
trainer = Trainer(logger=logger, max_epochs=100, gradient_clip_val=0.8)

# Qualitative results
#trainer.predict(model, d.val_dataloader())

# Train
trainer.fit(
    model, 
    train_dataloaders=d.train_dataloader(), 
    val_dataloaders=d.val_dataloader()
)
