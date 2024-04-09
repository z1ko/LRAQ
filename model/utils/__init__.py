from .criterion import MeanOverFramesAccuracy, F1Score, EditDistance
import torch

def calculate_metrics(predictions, targets, prefix='val'):

    mof = MeanOverFramesAccuracy()
    f1 = F1Score()
    edit = EditDistance(True)

    result = { 'mof': mof(predictions, targets), 'edit': edit(predictions, targets) }
    result.update(f1(predictions, targets))
    result = { f'{prefix}/{key}': val for key,val in result.items() }
    return result