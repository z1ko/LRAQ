from yacs.config import CfgNode
_C = CfgNode()

# ==================================================
# TRAINING

_C.TRAIN = CfgNode()

# [assembly101, ...]
_C.TRAIN.DATASET = 'assembly101'
# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 1
# the labels to ignore during evaluation
_C.TRAIN.IGNORE_LABELS = [-100]

# ==================================================
# MODEL

_C.MODEL = CfgNode()

# number of classes
_C.MODEL.NUM_CLASSES = 202
# the dimension of extracted features, I3D: 2048
_C.MODEL.INPUT_DIM = 2048
# ignore idx for cross entropy, the value to be used for padding targets
_C.MODEL.PAD_IGNORE_IDX = -100
# fraction of mse loss contributes to total loss
_C.MODEL.MSE_LOSS_FRACTION = 0.15
# mse loss clamp/clip value
_C.MODEL.MSE_LOSS_CLIP_VAL = 16.0


def get_config():
    return _C.clone()