import torch
import torch.nn as nn
from timm.models import VisionTransformer

class VisionEncoder(nn.Module):
    def __init__(self, args):
        super(VisionEncoder, self).__init__()

        self.args = args
        self.config = args.dataset_config["VisionEncoder"]
        raise NotImplementedError("Vision Encoder not implemented yet.")

    def forward(self):        
        pass