import torch
import torch.nn as nn

from models.VisionModules import VisionEncoder
from models.LanguageModules import LanguageEncoder
import torch.nn.functional as F

class Codigen(nn.Module):
    
    def __init__(self, args):
        super(Codigen, self).__init__()
        self.args = args
        self.config = args.dataset_config["Codigen"]
        
        self.place_holder = nn.Linear(1, 1)
        
        self.vis_encoder = VisionEncoder(args)
        self.lan_encoder = LanguageEncoder(args)
        # 154368
        self.sequential_network = nn.RNN(input_size=1536, hidden_size=768, num_layers=1, batch_first=True)
        
        self.fuse_network = nn.Sequential(
            nn.Linear(768, 4000),
            nn.ReLU(),
            nn.Linear(4000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 768),
        )
        self.__init__encoder()

        self.decoder = None # GPT?

    # THIS IS DIRECTLY COPIED FROM BaselineModules.py
    def __init__encoder(self):
        self.language_encoder = LanguageEncoder(self.args)
        self.ground_truth_encoder = LanguageEncoder(self.args)
        for param in self.ground_truth_encoder.parameters():
            param.requires_grad = False  # not update by gradient
            
        fused_dim = (self.args.dataset_config["seq_len"] - 1) * self.args.dataset_config["text_embed_dim"]
        self.fusion_module = nn.Linear(
            fused_dim, self.args.dataset_config["text_embed_dim"]
        )

    def forward_encoder(self, panels, texts):
        # encoder
        panel_embeddings = self.vis_encoder(panels) # output: [batch_size, seq_len, embedding_dim] embedding_dim = 768
        text_embeddings = self.lan_encoder(texts) # output: [batch_size, seq_len, embedding_dim] embedding_dim = 768 ?
        
    
        # Process embeddings
        text_embeddings = text_embeddings.mean(dim=2)  # output: [batch_size, seq_len, 100, embedding_dim]
        # Aggregate the token dimension, using max pooling across the token dimension (100)
        text_embeddings = text_embeddings.max(dim=2)[0]  # output: [batch_size, seq_len, embedding_dim]

        concat_embedding = torch.cat((panel_embeddings, text_embeddings), dim=2) # output: [batch_size, seq_len, embedding_dim * 2]
        
        # sequential network
        sequential_embedding, _ = self.sequential_network(concat_embedding)
        
        # fuse network
        fused_embeddings = self.fuse_network(sequential_embedding)
        fused_embeddings = self.pool(fused_embeddings)
        
        # Ground Truth Embeddings
        ground_truth_id = texts[:, -1, 0, :]
        ground_truth_mask = texts[:, -1, 1, :]
        gt_seq_embeddings = self.ground_truth_encoder(ground_truth_id, ground_truth_mask)
        gt_embeddings = self.pool(gt_seq_embeddings)

        return fused_embeddings, gt_embeddings

    # TODO: TO BE IMPLEMENTED
    def forward_decoder(self, embeddings):
        decoded_text = self.decoder(embeddings)
        return decoded_text
    
    def forward(self, panels=None, text=None, embeddings=None):
        if embeddings is None:
            """Encode only """
            embeddings, gt_embedding = self.forward_encoder(panels, text)
            return embeddings, gt_embedding, None, None
        else:
            return self.forward_decoder(embeddings)

    def pool(self, embedding):
        embedding_transposed = embedding.transpose(1, 2)
        # Apply max pooling over the sequence length dimension
        max_pooled = F.max_pool1d(embedding_transposed, kernel_size=embedding_transposed.size(-1))
        max_pooled = max_pooled.squeeze(-1)
        return max_pooled

    # def pool(self, embedding):
    #     # If embedding is more than 3D, flatten the additional dimensions
    #     if embedding.dim() > 3:
    #         embedding = embedding.view(embedding.size(0), embedding.size(1), -1)

    #     # Transpose to put seq_len as the last dimension
    #     embedding_transposed = embedding.transpose(1, 2)

    #     # Apply max pooling over the sequence length dimension
    #     max_pooled = F.max_pool1d(embedding_transposed, kernel_size=embedding_transposed.size(-1))

    #     # Squeeze the last dimension
    #     max_pooled = max_pooled.squeeze(-1)

    #     return max_pooled
