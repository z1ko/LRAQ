import matplotlib.pylab as plt
import statistics
import json

# function to add value labels
def add_labels(ax, x,y):
    for i in range(len(x)):
        yv = round(y[i], 3)
        ax.text(i,yv+0.1,yv, ha = 'center')

with open('data/exper_ablation_2.json', 'rt') as f:
    ablation = json.load(f)

# =================================================================================
# Tutte le ablazioni insieme, media sugli esercizi

fig, ax = plt.subplots()

xs = ablation['experiments'].keys()
ys = [ x['aggregated']['mean'] for x in ablation['experiments'].values()]
er = [ x['aggregated']['stdev'] for x in ablation['experiments'].values()]

ax.bar(xs, ys, color='blue')
ax.set_ylim([4.0, 6.5])
ax.axhline(min(ys), color='r')
fig.suptitle('Mean performance across all exercises', fontsize=16)

add_labels(ax, xs, ys)

fig.show()
plt.show()

# =================================================================================
# Ogni ablazione per tutti gli esercizi

fig, axs = plt.subplots(nrows=1, ncols=6, sharey=True)
for i, (experiment_name, data) in enumerate(ablation['experiments'].items()):

    # Visualizza tutti gli esercizi
    xs=[f'Ex{i}' for i in range(1,6)]
    ys=[x['folds_mean'] for x in data['exercises']]

    xs.append('mean')
    ys.append(float(statistics.mean(ys)))

    axs[i].bar(xs, ys, color=['blue']*5 + ['red'])
    axs[i].set_ylim([4.0, 8.0])
    axs[i].set_title(experiment_name)
    axs[i].legend()

fig.suptitle('Performance on all exercises', fontsize=16)
fig.show()
plt.show()

# =================================================================================
# Con e senza gMLP

fig, ax = plt.subplots()


xs=['LRU', 'GRU', 'LSTM']
ys = {
    'with_gmlp': [],
    'without_gmlp': []
}

for i, (experiment_name, data) in enumerate(ablation['experiments'].items()):
    if experiment_name.endswith('NoSpatial'):
        ys['without_gmlp'].append(data['aggregated']['mean'])
    else:
        ys['with_gmlp'].append(data['aggregated']['mean'])


width = 0.40  # the width of the bars
multiplier = 0
dist_base = list(range(len(xs)))

for attribute, measurement in ys.items():
    offset = width * multiplier
    dist = list(map(lambda x: x + offset, dist_base))
    rects = ax.bar(dist, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel('MAE')
ax.set_xticks(list(map(lambda x: x + width / 2, dist_base)), xs)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(4.0, 6.5)

fig.show()
plt.show()
