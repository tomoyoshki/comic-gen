import torch.nn as nn

class CodigenLoss(nn.Module):
    def __init__(self, args):
        super(CodigenLoss, self).__init__()
        self.args = args
        self.framework_config = self.args.dataset_config["Codigen"]
        raise NotImplementedError("Codigen Loss not implemented yet")

    def forward(self):
        pass

# add more losses here
# similarity loss, embedding loss, contrastive loss?
# Contrastive loss reference: simclr https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
# embedding similarity loss: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html