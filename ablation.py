from datetime import datetime
from itertools import chain, combinations, product
from train_complete import save_to_disk, train_complete


# All types of temporal methods available in the study
TEMPORAL_METHODS = ['GRU', 'LSTM', 'LRU']
# All type of spatial methods available in the study
SPATIAL_METHODS = ['gMLP', 'NoSpatial']
# All meta-options available in the study
OPTIONS = [] #['NoConv']


# Network descriptor, equal for all tests
NETWORK = {

    #LRU specific
    'lru_phase': 6.283185307179586,
    'lru_radius_min': 0.80,
    'lru_radius_max': 0.99,

    # Network architecture
    'model_dim': 96,
    'joint_features': 6,
    'temporal_state_dim': 96,
    'temporal_layers': 4,
    'spatial_layers': 8,
    'dropout': 0.7,

    # Dataset and learning parameters
    'joint_count': 19,
    'maximum_quality': 50.0,
    'batch_size': 12,
    "epochs": 200,
    'learning_rate': 0.002,
    'weight_decay': 0.001,
    'scheduler_step': 100,
}


def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def create_network_string(temporal_method, spatial_method, options):
    return f'{temporal_method}+{spatial_method}{"+" if len(options) != 0 else ""}{"+".join(options)}'


# Ablation executor
if __name__ == '__main__':

    time = datetime.now()
    results = {
        'configuration_base': NETWORK,
        'experiments': {}
    }

    for options in powerset(OPTIONS):
        for temporal_method, spatial_method in product(TEMPORAL_METHODS, SPATIAL_METHODS):

            # Create string of the experiment
            string = create_network_string(temporal_method, spatial_method, options)
            print('====================================================================================')
            print('RUNNING EXPERIMENT WITH CONFIGURATION: ', string)
                
            opts = {
                'temporal_method': temporal_method,
                'spatial_method': spatial_method,
                'no_conv': 'NoConv' in options,
                **NETWORK
            }

            # Run X experiments and get final loss
            train_result = train_complete(f'ablation_{string}', time, opts)
                
            print(f"RESULT FOR CONFIGURATION {string}: {train_result['aggregated']['mean']}")
            print('====================================================================================')

            # Save experiment data
            results['experiments'][string] = train_result
            save_to_disk('ablation', time, results)
    
    # Save complete data to file
    save_to_disk('ablation', time, results)

    #def get_exercise_data(data, exercise):
    #    results = list(filter(lambda e: e['exercise'] == exercise, data))
    #    names = list(map(lambda e: create_network_string(e['temporal_method'], e['spatial_method'], e['options']), results))
    #    losses = list(map(lambda e: e['best_loss_mean'], results))
    #    errors = list(map(lambda e: e['best_loss_std'], results))
    #    return names, losses, errors
    #
    ## Show results
    #fig, axs = plt.subplots(1, 5, sharey=True)
    #for i in range(len(1, 6)):
    #    names, values, errors = get_exercise_data(results['experiments'], i+1)
    #    axs[i].title.set_text(f'Exercise {i+1}')
    #    axs[i].errorbar(values, names, xerr=errors, linestyle='None', marker='^')
    #    axs[i].legend()
    #
    #
    #fig.suptitle(f'MAE for each configuration and exercise')
    #plt.show()
