import torch.nn as nn


class Codigen(nn.Module):
    def __init__(self, args):
        super(Codigen, self).__init__()

        self.args = args
        self.config = args.dataset_config["Codigen"]
        raise NotImplementedError("Codigen not implemented yet")

    def forward(self):        
        pass