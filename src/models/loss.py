import torch
import torch.nn as nn

from models.loss_utils import mask_correlated_samples

class GeneralLoss(nn.Module):
    def __init__(self, args):
        super(GeneralLoss, self).__init__()
        self.args = args
        self.framework_config = self.args.dataset_config["Codigen"]
        self.temperature = self.framework_config["temperature"]
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward_similarity(self, z_i, z_j):
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        # Calculate similarity
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        negative_masks = mask_correlated_samples(batch_size).to(self.args.device)
        negative_samples = sim[negative_masks].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss
        
    def forward(self, predicted, gt):
        if self.args.stage in {"encode"}:
            return self.forward_similarity(predicted, gt)
        else:
            return self.criterion(predicted.float(), gt.float())

# add more losses here
# similarity loss, embedding loss, contrastive loss?
# Contrastive loss reference: simclr https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
# embedding similarity loss: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html