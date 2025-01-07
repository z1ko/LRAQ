import argparse
import json
import tempfile
import datetime
import pickle

from model.regression import LRGA
from model.utils.transform import Compose, JointDifference
from dataset.kimore import KiMoReDataModuleFolded

from lightning.pytorch import Trainer, Callback
from statistics import mean, stdev
from ray import train, tune

from train_complete import save_to_disk, train_complete

# Complete training of the model
def objective(config, time, data):

    train_result = train_complete(f'sweep', time, config, data=data, log=False)
    train.report({
        'aggregated_mean': train_result['aggregated']['mean'],
        'parameters': train_result['parameters'],
        'ex1_mean': train_result['exercises'][0]['folds_mean'],
        'ex2_mean': train_result['exercises'][1]['folds_mean'],
        'ex3_mean': train_result['exercises'][2]['folds_mean'],
        'ex4_mean': train_result['exercises'][3]['folds_mean'],
        'ex5_mean': train_result['exercises'][4]['folds_mean'],
    })


# Load all datasets
with open('data/processed/kimore_kfold.pickle', 'rb') as f:
    data = pickle.load(f)

# Here we define our configuration for the trials
time = datetime.datetime.now()
trainable_with_resources = tune.with_resources(objective, {'gpu': 1.0})
tuner = tune.Tuner(
    tune.with_parameters(trainable_with_resources, time=time, data=data),
    tune_config=tune.TuneConfig(num_samples=30),
    param_space={

        # What we want to explore
        'model_dim': tune.qrandint(32, 256, 32),
        'temporal_state_dim': tune.qrandint(32, 256, 32),
        'temporal_layers': tune.qrandint(4, 10, 2),
        'spatial_layers': tune.qrandint(4, 10, 2),
        'dropout': tune.quniform(0.2, 0.8, 0.05),
        'batch_size': tune.qrandint(6, 16, 2),
        'learning_rate': tune.quniform(0.0005, 0.002, 0.0005),
        'weight_decay': tune.choice([0.0001, 0.001]),

        # Fixed
        'maximum_quality': 50.0,
        'joint_count': 19,
        'joint_features': 6,
        'scheduler_step': 100,
        'temporal_method': 'GRU',
        'spatial_method': 'gMLP',
        'no_conv': False,
        'epochs': 200
    }
)

results = tuner.fit()
print(
    results.get_best_result(
        metric="aggregated_mean", 
        mode="min"
    ).config
)

# Store result to file
with open(f'sweep_result_{time}.pickle', 'wb') as f:
    pickle.dump(results, f)
