import matplotlib.pyplot as plt
import json

def create_network_string(temporal_method, spatial_method, options):
    return f'{temporal_method}+{spatial_method}{"+" if len(options) != 0 else ""}{"+".join(options)}'

def create_network_string_impl(e):
    return create_network_string(e['temporal_method'], e['spatial_method'], e['options'])

def filter_data(data, filter_fn, names_fn=create_network_string_impl):
    results = list(filter(filter_fn, data))
    names = list(map(lambda e: names_fn(e), results))
    losses = list(map(lambda e: e['best_loss_mean'], results))
    errors = list(map(lambda e: e['best_loss_std'], results))
    maxes = list(map(lambda e: e['best_loss'], results))
    return names, losses, errors, maxes


def plot_exercise(ax, data, exercise):
    names, values, errors, maxes = filter_data(data, lambda e: e['exercise'] == exercise)
    
    colors = []
    for i in range(len(names)):
        c = ['red', 'green', 'blue', 'cyan', 'magenta'][int(i/2)]
        colors.append(c)

    ax.title.set_text(f'Exercise {exercise}')
    ax.barh(names, values, xerr=errors, alpha=0.5, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), color=colors, zorder=1)
    ax.scatter(maxes, names, marker='*', sizes=[300]*len(maxes), zorder=2, color=["yellow"]*len(maxes))
    ax.set_xlim(1.0, 7.0)
    ax.legend()

def show_spatial_vs_no_spatial(data):

    names, spatial_values, _, spatial_best = filter_data(data, lambda e: e['spatial_method'] != 'NoSpatial', names_fn=lambda e: e['temporal_method'])
    names, no_spatial_values, _, no_spatial_best = filter_data(data, lambda e: e['spatial_method'] == 'NoSpatial', names_fn=lambda e: e['temporal_method'])

    maker_sizes = [60]*len(spatial_values)
    fig, axs = plt.subplots(1, 1, sharey=True)
    axs.scatter(no_spatial_best, names, label='NoSpatial', linestyle='None', marker='*', sizes=maker_sizes)
    axs.scatter(spatial_best, names, label='gMLP', linestyle='None', marker='+', sizes=maker_sizes)
    axs.legend()

    fig.suptitle(f'Spatial vs NoSpatial (lower is better)')
    plt.show()

def show_temporal_methods(data, methods):
    values = [0] * len(methods)
    for i, method in enumerate(methods): #['LRU', 'LSTM', 'GRU', 'RNN']:
        _, spatial_values, _, _ = filter_data(data, lambda e: e['temporal_method'] == method and e['spatial_method'] != 'NoSpatial')
        _, no_spatial_values, _, _ = filter_data(data, lambda e: e['temporal_method'] == method and e['spatial_method'] == 'NoSpatial')
        values[i] = (spatial_values[0] + no_spatial_values[0]) * 0.5

    fig, axs = plt.subplots(1, 1, sharey=True)
    axs.bar(methods, values)
    axs.set_ylim(3.0, 7.0)

    fig.suptitle(f'Different temporal methods (lower is better)')
    plt.show()


# kimore_ex1 & 1.850000 & 2.774013 & 4.057242 & 6.152314 & 5.291101 \\
# kimore_ex2 & 1.570000 & 3.878395 & 4.475701 & 4.524518 & 4.729095 \\
# kimore_ex3 & 0.958000 & 3.573795 & 3.228595 & 2.977246 & 3.013976 \\
# kimore_ex4 & 1.510000 & 2.442373 & 3.653577 & 3.487373 & 3.619473 \\
# mean & 1.472000 & 3.167144 & 3.853779 & 4.285363 & 4.163411 \\

def show_lru_vs_sota():
    model = ['ActionQ', 'RIGCN', 'SGN', 'VAGCN', 'STGCN']
    values = [3.230, 3.167, 3.854, 4.285, 4.163]
    fig, axs = plt.subplots(1, 1, sharey=True)
    fig.suptitle(f'ActionQ vs others. (lower is better, best model instance available)')
    axs.bar(model, values) #, color = ['red', 'green', 'blue', 'orange', 'cyan'])
    axs.set_ylim(3.0, 5.0)
    plt.show()

# Read results data
FILEPATH = 'ablations/log_20241212201011.json' #'ablations/important/log_20241212161815.json' 
with open(FILEPATH, 'r', encoding='utf-8') as f:
    results = json.load(f)

show_temporal_methods(results['experiments'], ['LRU', 'LSTM', 'GRU', 'RNN'])

fig, axs = plt.subplots(1, 1, sharey=True)
plot_exercise(axs, results['experiments'], 1)
plt.show()

show_spatial_vs_no_spatial(results['experiments'])