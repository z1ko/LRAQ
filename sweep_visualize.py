import pickle
import matplotlib.pyplot as plt
import json
import glob

from ray import train, tune

with open('sweep_result_2025-01-06 12:01:02.311333.pickle', 'rb') as f:
    results = pickle.load(f)

best_result = results.get_best_result(metric="aggregated_mean", mode="min")
print(best_result.config)


