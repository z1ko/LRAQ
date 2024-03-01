
import matplotlib.pyplot as plt
import wandb

from lightning.pytorch import Trainer
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.loggers import WandbLogger

from model.regression import LRGA
from model.gtt import G2TAQ

from model.utils.args import base_arg_parser
from model.utils.transform import Compose, JointDifference
from dataset.kimore import KiMoReDataModule

parser = base_arg_parser(LRGA, KiMoReDataModule)
opts = parser.parse_args()
opts = vars(opts)
print(opts)

d = KiMoReDataModule(
    data_dir='data/processed/kimore', 
    exercise=2, 
    window_size=opts['window_size'],
    window_delta=opts['window_size'],
    batch_size=opts['batch_size'],
    leave_one_out=None,
    transform=Compose([
        JointDifference(),
    ])
)
d.setup()

# Bello bello
model = LRGA(
    joint_count=19,
    maximum_quality=50,
    **opts
)

#summary = ModelSummary(model, max_depth=-1)
#print(summary)

logger = None
if opts['dashboard']:
    logger = WandbLogger(log_model='all', save_dir='runs/')
    logger.log_hyperparams(opts)


trainer = Trainer(max_epochs=opts['epochs'], logger=logger)
trainer.fit(
    model, 
    train_dataloaders=d.train_dataloader(), 
    val_dataloaders=d.val_dataloader()
)

