#import matplotlib.pyplot as plt
#import wandb

from lightning.pytorch import Trainer
#from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.loggers import WandbLogger

#from model.utils.args import base_arg_parser
from dataset.assembly import AssemblyDataModule
from model.tas import TemporalActionSegmentation

#parser = base_arg_parser()
#opts = parser.parse_args()
#print(vars(opts))

d = AssemblyDataModule(
    'data/processed/assembly', 
    batch_size=6,
    window_size=5000
)
d.setup()

model = TemporalActionSegmentation(
    model_dim=64,
    learning_rate=0.001,
    weight_decay=0.005,
    scheduler_step=200,
    joint_count=2*21,
    joint_features=3,
    temporal_state_dim=128,
    spatial_state_dim=64,
    temporal_layers=3,
    spatial_layers=3
)

#logger = None
#if opts.dashboard:
#    logger = WandbLogger(save_dir='runs/')
#    logger.log_hyperparams(opts)

trainer = Trainer(max_epochs=20, detect_anomaly=True, overfit_batches=0.1)
trainer.fit(
    model, 
    train_dataloaders=d.train_dataloader(), 
    val_dataloaders=d.val_dataloader()
)
