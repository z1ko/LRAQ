import torch.nn as nn
import torch

from model.utils.args import base_arg_parser
from model.regression import LRGA

parser = base_arg_parser(LRGA)
opts = parser.parse_args()
opts = vars(opts)
print(opts)

# Model optimized for inference
class AQInference(nn.Module):
    def __init__():
        super().__init__()

        self.embed = nn.Linear(6, 32)
        self.


        pass


model = LRGA(joint_count=19, maximum_quality=50, batch_size=1, **opts)
static_input = torch.randn(1, 1, 19, 6)
onnx = torch.onnx.dynamo_export(model, static_input)
onnx.save('actionq.onnx')
