import wandb
import argparse
import yaml

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from model.regression import LRGA
from model.utils.transform import Compose, JointDifference
from dataset.kimore import KiMoReDataModule

dataset = None

def train_model():
    config = wandb.config
    model = LRGA(
        joint_count=19,
        maximum_quality=50,
        **config
    )

    d = KiMoReDataModule(
        data_dir='data/processed/kimore',
        exercise=1,
        window_size=config.window_size,
        window_delta=config.window_size,
        batch_size=config.batch_size,
        leave_one_out=None,
        transform=Compose([
            JointDifference(),
        ])
    )
    d.setup()

    logger = WandbLogger(log_model=True, save_dir='runs/')
    logger.log_hyperparams(config)

    trainer = Trainer(max_epochs=config.epochs, logger=logger)
    trainer.fit(
        model, 
        train_dataloaders=d.train_dataloader(),
        val_dataloaders=d.val_dataloader()
    )

def main(args):
    with open(args.configuration) as f:
        sweep_config = yaml.load(f, Loader=yaml.Loader)

    sweep_id = wandb.sweep(sweep=sweep_config, project="LRGA")
    wandb.agent(sweep_id, train_model, count=args.sweep_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', type=str, default='config/basic.yml')
    parser.add_argument('sweep_count', type=int, default=100)
    args = parser.parse_args()
    main(args)
