import argparse
import json
import tempfile

from model.regression import LRGA
from model.utils.transform import Compose, JointDifference
from dataset.kimore import KiMoReDataModuleFolded

from lightning.pytorch import Trainer, Callback
from statistics import mean, stdev
from ray import train, tune


EXERCISES = [1,2,3,4,5]
FOLDS = [0,1,2,3,4]


class BestValidationCallback(Callback):
    def __init__(self, log_best):
        super().__init__()
        self.best_loss = 10000.0
        self.log_best = log_best

    def on_validation_epoch_end(self, trainer, module):
        mean_batches_loss = mean(module.validation_losses)
        self.best_loss = min(mean_batches_loss, self.best_loss)
        if self.log_best:
            self.log("validation/best-loss-mae", self.best_loss, prog_bar=True)


# Complete training of the model
def objective(config, data):

    # Train model
    callback = BestValidationCallback(log_best=True)
    trainer = Trainer(max_epochs=config['epochs'], callbacks=[callback])
    trainer.fit(
        model=LRGA(joint_count=19, maximum_quality=50, **config),
        train_dataloaders=data.train_dataloader(), 
        val_dataloaders=data.val_dataloader()
    )

    # Send train result to Tune
    train.report({
        'best_loss': callback.best_loss,
        'memory': 0.0
    })


# Keep dataset here
dataset = KiMoReDataModuleFolded(
    filepath='data/processed/kimore_kfold.pickle',
    batch_size=10,
    transform=Compose([JointDifference()]),
    exercise=1,
    fold=0,
)
dataset.setup()

# Here we define our configuration for the trials
trainable_with_resources = tune.with_resources(objective, {'gpu': 0.25})
tuner = tune.Tuner(
    tune.with_parameters(trainable_with_resources, data=dataset),
    param_space={

        # What we want to explore
        'model_dim': tune.grid_search([32, 64, 128]),
        'temporal_state_dim': tune.grid_search([32, 64, 128]),
        'temporal_layers': tune.grid_search([2, 4, 6]),
        'spatial_layers': tune.grid_search([2, 4, 6]),

        # Fixed
        'learning_rate': 0.001,
        'weight_decay': 0.001,
        'dropout': 0.3,
        'joint_features': 6,
        'scheduler_step': 300,
        'temporal_method': 'LRU',
        'spatial_method': 'gMLP',
        'batch_size': 10,
        'no_conv': False,
        'epochs': 1
    }
)

results = tuner.fit()
print(results.get_best_result(metric="best_loss", mode="min").config)
