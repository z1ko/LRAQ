# Train the network to completion, on all exercises and all folds

from statistics import stdev, mean
from datetime import datetime

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback

from model.regression import LRGA
from model.utils.args import base_arg_parser
from model.utils.transform import Compose, JointDifference

from dataset.kimore import KiMoReDataModuleFolded

import json

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


def save_to_disk(string, time, results):
    with open(f'training/exper_{string}_{time.strftime("%Y%m%d%H%M%S")}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_all_folds(exercise, opts, data):

    folds_loss = []
    for fold in FOLDS:

        # Load dataset
        dataset = KiMoReDataModuleFolded(
            filepath='data/processed/kimore_kfold.pickle',
            data=data,
            batch_size=opts['batch_size'],
            transform=Compose([JointDifference()]),
            exercise=exercise,
            fold=fold,
        )
        dataset.setup()
            
        #model=LRGA(joint_count=19, maximum_quality=50.0, **opts)
        model=LRGA(**opts)
        parameters = count_model_parameters(model)

        # Train model
        callback = BestValidationCallback(log_best=True)
        trainer = Trainer(max_epochs=opts['epochs'], callbacks=[callback])
        trainer.fit(
            model=model,
            train_dataloaders=dataset.train_dataloader(), 
            val_dataloaders=dataset.val_dataloader()
        )
            
        # Save best model in this fold
        folds_loss.append(callback.best_loss)

    folds_mean = mean(folds_loss)
    folds_stdev = stdev(folds_loss)
    folds_best = min(folds_loss)

    return {
        'exercise': exercise,
        'parameters': parameters,
        'folds_loss': folds_loss,
        'folds_mean': folds_mean,
        'folds_stdev': folds_stdev,
        'folds_best': folds_best
    }


# Train the model with a specific configuration, on a subset of exercises, on all folds
def train_complete(string, time, opts, log=True, data=None):

    results = {
        'configuration': opts,
        'parameters': 0,
        'exercises': [],
        'aggregated': {}
    }

    for exercise in EXERCISES:
        loss = train_all_folds(exercise, opts, data)
        print(loss)

        results['parameters'] = loss['parameters']
        results['exercises'].append(loss)
        
        if log:
            save_to_disk(string, time, results)

    exercises_loss = [exercise['folds_mean'] for exercise in results['exercises']]
    results['aggregated']['stdev'] = stdev(exercises_loss)
    results['aggregated']['mean'] = mean(exercises_loss)

    if log:
        save_to_disk(string, time, results)

    return results


if __name__ == '__main__':

    parser = base_arg_parser(LRGA, KiMoReDataModuleFolded)
    opts = parser.parse_args()
    opts = vars(opts)
    print(opts)

    time = datetime.now()
    results = train_complete('X', time, opts)
    print(results)
