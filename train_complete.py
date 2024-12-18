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

def save_to_disk(time, results):
    with open(f'training/exper_{time.strftime("%Y%m%d%H%M%S")}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

parser = base_arg_parser(LRGA, KiMoReDataModuleFolded)
opts = parser.parse_args()
opts = vars(opts)
print(opts)


# Train the model with a specific configuration, on a subset of exercises, on all folds
#def train(opts, )


results = {
    'options': opts,
    'exercises': [],
    'aggregated': {}
}

time = datetime.now()

exercises_loss = []
for exercise in EXERCISES:

    folds_loss = []
    for fold in FOLDS:

        # Load dataset
        dataset = KiMoReDataModuleFolded(
            filepath='data/processed/kimore_kfold.pickle',
            batch_size=opts['batch_size'],
            transform=Compose([JointDifference()]),
            exercise=exercise,
            fold=fold,
        )
        dataset.setup()
        
        # Train model
        callback = BestValidationCallback(log_best=True)
        trainer = Trainer(max_epochs=opts['epochs'], callbacks=[callback])
        trainer.fit(
            model=LRGA(joint_count=19, maximum_quality=50, **opts),
            train_dataloaders=dataset.train_dataloader(), 
            val_dataloaders=dataset.val_dataloader()
        )
        
        # Save best model in this fold
        folds_loss.append(callback.best_loss)

    folds_mean = mean(folds_loss)
    folds_stdev = stdev(folds_loss)
    folds_best = min(folds_loss)
    
    exercises_loss.append(folds_mean)
    results['exercises'].append({
        'exercise': exercise,
        'folds_loss': folds_loss,
        'folds_mean': folds_mean,
        'folds_stdev': folds_stdev,
        'folds_best': folds_best
    })

    save_to_disk(time, results)

# Store aggregated data
results['aggregated']['mean'] = mean(exercises_loss)
results['aggregated']['stdev'] = stdev(exercises_loss)
save_to_disk(time, results)
