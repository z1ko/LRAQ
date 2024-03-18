import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as ein
import numpy as np


#def narrow_gaussian(x, ell):
#    return torch.exp(-0.5 * (x / ell) ** 2)
#
## The spice must flow
#def approx_count_nonzero(x, ell=1e-3):
#    return len(x) - narrow_gaussian(x, ell).sum(dim=-1)
#
#def mean_over_frames(input, target):
#    assert(input.shape == target.shape)
#    T = input.shape[1]
#
#    difference = input - target
#    zeros = T - approx_count_nonzero(difference)
#    mof = zeros / T
#    return mof.mean()

# NOTE: used for training
class CEplusMSE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """

    def __init__(self, opts):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.alpha = opts.loss_alpha
        self.classes = opts.classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        :param logits: [batch_size, classes, seq_len]
        :param targets: [batch_size, seq_len]
        """

        loss_dict = { 'loss': 0.0, 'loss_ce': 0.0, 'loss_mse': 0.0 }
        for p in logits:

            loss_dict['loss_ce'] += self.ce(
                ein.rearrange(p, "b classes seq_len -> (b seq_len) classes"),
                ein.rearrange(targets, "b seq_len -> (b seq_len)")
            )

            loss_dict['loss_mse'] += torch.mean(torch.clamp(self.mse(
                F.log_softmax(p[:, :, 1:], dim=1),
                F.log_softmax(p.detach()[:, :, :-1], dim=1)
            )))

        loss_dict['loss'] = loss_dict['loss_ce'] + self.alpha * loss_dict['loss_mse']
        return loss_dict

class MeanOverFramesAccuracy:
    def __init__(self):
        self.total = 0
        self.correct = 0

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        :param predictions: [batch_size, seq_len]
        :param targets: [batch_size, seq_len]
        """

        predictions, targets = np.array(predictions), np.array(targets)
        total = predictions.shape[-1]
        correct = (predictions == targets).sum()
        result = correct / total if total != 0 else 0

        self.total += total
        self.correct += correct
        return result


class F1Score:
    def __init__(self, classes, overlaps = [0.1, 0.25, 0.5]):
        self.overlaps = overlaps
        self.classes = classes

        self.tp = np.zeros((len(overlaps), classes))
        self.fp = np.zeros((len(overlaps), classes))
        self.fn = np.zeros((len(overlaps), classes))

    def forward(self, predictions, targets):
        """
        :param predictions: [batch_size, seq_len]
        :param targets: [batch_size, seq_len]
        """

        result = {}

        predictions, targets = np.array(predictions), np.array(targets)
        for p, t in zip(predictions, targets):
            result = {}

            for i, overlap in enumerate(self.overlaps):
                tp, fp, fn = self.f_score(
                    p.tolist(),
                    t.tolist(),
                    overlap
                )
                
                self.tp[i] += tp
                self.fp[i] += fp
                self.fn[i] += fn

                f1 = self.get_f1_score(tp, fp, fn)
                result[f'F1@{overlap*100}'] = f1

        return result

    @staticmethod
    def f_score(predictions, targets, overlap):
        pass

    @staticmethod
    def get_f1_score(tp, fp, fn):
        if tp + fp != 0.0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0.0
            recall = 0.0
        
        if precision + recall != 0.0:
            return 2.0 * (precision * recall) / (precision + recall)
        else:
            return 0.0

