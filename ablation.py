
from datetime import datetime
from copy import deepcopy
from itertools import chain, combinations, product
from statistics import stdev, mean
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from dataset.kimore import KiMoReDataModule
from model.utils.transform import Compose, JointDifference
from model.regression import LRGA

import matplotlib.pyplot as plt
import json


# All types of temporal methods available in the study
TEMPORAL_METHODS = ['LRU', 'RNN', 'GRU', 'LSTM']
# All type of spatial methods available in the study
SPATIAL_METHODS = ['gMLP', 'NoSpatial']
# All meta-options available in the study
OPTIONS = [] #['NoConv']

# The exercises present in the study
EXERCISES =  [1, 2, 3, 4]

# Number of tests for each configuration
CONFIG_TEST_COUNT = 10
# Number of epochs for each test
CONFIG_TEST_EPOCS = 400

# Network descriptor, equal for all tests
NETWORK = {
    'dashboard': False,
    'lru_phase': 6.283185307179586,
    'lru_radius_min': 0.80,
    'lru_radius_max': 0.99,
    'model_dim': 32,
    'joint_features': 6,
    'temporal_state_dim': 48,
    'temporal_layers': 2,
    'spatial_layers': 2,
    'dropout': 0.3,
    'use_convolution': True,
    'use_spatial': False,
    'window_size': 200,
    'window_delta': 200,
    'batch_size': 10,
    'dataset': None,
    'learning_rate': 0.001,
    'weight_decay': 0.001,
    'scheduler_step': 300,
    'checkpoint': False,
    'checkpoint_epochs': 5,
    'joint_count': 19,
    'maximum_quality': 50.0
}


def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


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


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


# Run a single experiment with the provided options
def run_experiment_set(opts, count, epochs, data):

    best_losses = []
    for i in range(count):
        callback = BestValidationCallback(log_best=True)
        trainer = Trainer(max_epochs=epochs, callbacks=[callback])
        model = LRGA(**opts)
        parameters = count_model_parameters(model)
        trainer.fit(
            model, 
            train_dataloaders=data.train_dataloader(), 
            val_dataloaders=data.val_dataloader()
        )
        best_losses.append(callback.best_loss)

    return min(best_losses), mean(best_losses), stdev(best_losses), parameters


def create_network_string(temporal_method, spatial_method, options):
    return f'{temporal_method}+{spatial_method}{"+" if len(options) != 0 else ""}{"+".join(options)}'


def save_to_disk(time, results):
    with open(f'ablations/log_{time.strftime("%Y%m%d%H%M%S")}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# Ablation executor
if __name__ == '__main__':

    time = datetime.now()
    results = {
        'network_configuration': NETWORK,
        'experiments': []
    }

    for exercise in EXERCISES:

        # Load correct dataset at the beginning, this way all the models have the same initial condition
        data = KiMoReDataModule(
            data_dir='data/processed/kimore',
            exercise=exercise,
            window_size=NETWORK['window_size'],
            window_delta=NETWORK['window_size'],
            batch_size=NETWORK['batch_size'],
            leave_one_out=None,
            transform=Compose([
                JointDifference(),
            ])
        )
        data.setup()

        for options in powerset(OPTIONS):
            for temporal_method, spatial_method in product(TEMPORAL_METHODS, SPATIAL_METHODS):

                # Create string of the experiment
                experiment_string = f'Ex{exercise}+{create_network_string(temporal_method, spatial_method, options)}'
                print('====================================================================================')
                print('RUNNING EXPERIMENT WITH CONFIGURATION: ', experiment_string)
                
                opts = {
                    'temporal_method': temporal_method,
                    'spatial_method': spatial_method,
                    'no_conv': 'NoConv' in options,
                    **NETWORK
                }

                # Run X experiments and get final loss
                loss_best, loss_mean, loss_std, parameters = run_experiment_set(opts, CONFIG_TEST_COUNT, CONFIG_TEST_EPOCS, data)
                
                print(f'RESULT FOR CONFIGURATION {experiment_string}: {loss_mean:.4f} +- {loss_std:.4f}')
                print('====================================================================================')

                # Save experiment data
                results['experiments'].append({ 
                    'experiment': experiment_string,
                    'exercise': exercise,
                    'temporal_method': temporal_method,
                    'spatial_method': spatial_method,
                    'options': [*options],
                    'best_loss_mean': loss_mean, 
                    'best_loss_std': loss_std,
                    'best_loss': loss_best,
                    'parameters': parameters
                })

                # Save partial data
                save_to_disk(time, results)

        del data
    
    # Save complete data to file
    save_to_disk(time, results)

    def get_exercise_data(data, exercise):
        results = list(filter(lambda e: e['exercise'] == exercise, data))
        names = list(map(lambda e: create_network_string(e['temporal_method'], e['spatial_method'], e['options']), results))
        losses = list(map(lambda e: e['best_loss_mean'], results))
        errors = list(map(lambda e: e['best_loss_std'], results))
        return names, losses, errors

    # Show results
    fig, axs = plt.subplots(1, 5, sharey=True)
    for i in range(len(EXERCISES)):
        names, values, errors = get_exercise_data(results['experiments'], i+1)
        axs[i].title.set_text(f'Exercise {i+1}')
        axs[i].errorbar(values, names, xerr=errors, linestyle='None', marker='^')
        axs[i].legend()


    fig.suptitle(f'MAE of {CONFIG_TEST_COUNT} different models for each configuration and exercise')
    plt.show()