import matplotlib.pylab as plt
import statistics
import json

# function to add value labels
def add_labels(ax, x,y):
    for i in range(len(x)):
        yv = round(y[i], 3)
        ax.text(i,yv+0.1,yv, ha = 'center')

with open('data/competitors.json', 'rt') as f:
    competitors = json.load(f)

# =================================================================================
# Tutte i modelli insieme, media sugli esercizi

fig, ax = plt.subplots()

xs = competitors.keys()
ys = [ x['aggregated']['mean'] for x in competitors.values()]

ax.bar(xs, ys, color='blue')
ax.set_ylim([4.0, 6.5])
ax.axhline(min(ys), color='r')
fig.suptitle('Mean performance of ActionQ and competitors across all exercises', fontsize=16)

#add_labels(ax, xs, ys)

fig.show()
plt.show()

# =================================================================================
# Ogni modello per tutti gli esercizi

fig, axs = plt.subplots(nrows=1, ncols=5, sharey=True)
for i, (competitor_name, data) in enumerate(competitors.items()):

    # Visualizza tutti gli esercizi
    xs=[f'Ex{i}' for i in range(1,6)]
    ys=[x['folds_mean'] for x in data['exercises']]

    xs.append('mean')
    ys.append(float(statistics.mean(ys)))

    axs[i].bar(xs, ys, color=['blue']*5 + ['red'])
    axs[i].set_ylim([3.5, 9.0])
    axs[i].set_title(competitor_name)
    axs[i].legend()

fig.suptitle('Performance on all exercises of ActionQ and competitors', fontsize=16)
fig.show()
plt.show()

# =================================================================================
# Ogni modello per tutti gli esercizi

fig, axs = plt.subplots(nrows=1, ncols=5, sharey=True)
for exercise in range(0,5):

    xs = competitors.keys()
    ys = [ c['exercises'][exercise] for c in competitors.values()]

    # Visualizza tutti gli esercizi
    xs=[f'Ex{i}' for i in range(1,6)]
    ys=[x['folds_mean'] for x in data['exercises']]

    xs.append('mean')
    ys.append(float(statistics.mean(ys)))

    axs[i].bar(xs, ys, color=['blue']*5 + ['red'])
    axs[i].set_ylim([3.5, 9.0])
    axs[i].set_title(competitor_name)
    axs[i].legend()

fig.suptitle('Performance on all exercises of ActionQ and competitors', fontsize=16)
fig.show()
plt.show()