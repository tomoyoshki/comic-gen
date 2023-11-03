import torch
import torch.nn as nn
from timm.models import VisionTransformer

class VisionEncoder(nn.Module):
    def __init__(self, args):
        super(VisionEncoder, self).__init__()

        self.args = args
        self.config = args.dataset_config["VisionEncoder"]
        
        # self.model = VisionTransformer()

    def forward(self):        
        raise NotImplementedError("Vision Encoder forward pass not implemented yet.")
        pass