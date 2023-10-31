import torch.nn as nn

class CodigenLoss(nn.Module):
    def __init__(self, args):
        super(CodigenLoss, self).__init__()
        self.args = args
        self.framework_config = self.args.dataset_config["Codigen"]
        raise NotImplementedError("Codigen Loss not implemented yet")

    def forward(self):
        pass

