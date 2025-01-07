import matplotlib.pyplot as plt
import argparse
import json
import glob

parser = argparse.ArgumentParser()
parser.add_argument('input_folder')
args = parser.parse_args()

experiments = []
for filepath in glob.glob(f'{args.input_folder}/*.json'):
    with open(filepath, 'rt') as f:
        experiments.append((filepath, json.load(f)))

print(experiments)

cols = list(map(lambda x: x[0].split('.')[0], experiments))
vals = list(map(lambda x: x[1]['aggregated']['mean'], experiments))

fig, ax = plt.subplots()
ax.bar(cols, vals)
ax.set_ylim([4.0, 7.0])
ax.tick_params(axis='x', labelrotation=45)

fig.savefig("test.png")




