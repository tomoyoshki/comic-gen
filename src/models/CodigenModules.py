import torch
import torch.nn as nn

from models.VisionModules import VisualEncoder
from models.LanguageModules import LanguageEncoder

class Codigen(nn.Module):
    
    def __init__(self, args):
        super(Codigen, self).__init__()

        self.args = args
        self.config = args.dataset_config["Codigen"]
        
        self.place_holder = nn.Linear(1, 1)
        
        self.vis_encoder = VisualEncoder(args)
        self.lan_encoder = LanguageEncoder(args)
        
        self.sequential_network = nn.RNN(input_size=1536, hidden_size=768, num_layers=1, batch_first=True)
        
        self.fuse_network = nn.Sequential(
            nn.Linear(768, 4000),
            nn.ReLU(),
            nn.Linear(4000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 768),
        )

        self.decoder = None # GPT?

    def forward_encoder(self, panels, texts):
        # encoder
        panel_embeddings = self.vis_encoder(panels) # output: [batch_size, seq_len, embedding_dim] embedding_dim = 768
        text_embeddings = self.lan_encoder(texts) # output: [batch_size, seq_len, embedding_dim] embedding_dim = 768 ?
        
        concat_embedding = torch.cat((panel_embeddings, text_embeddings), dim=2) # output: [batch_size, seq_len, embedding_dim * 2]
        
        # sequential network
        sequential_embedding, _ = self.sequential_network(concat_embedding)
        
        # fuse network
        fused_embeddings = self.fuse_network(sequential_embedding)
        
        return fused_embeddings

    def forward_decoder(self, embeddings):
        decoded_text = self.decoder(embeddings)
        return decoded_text
    
    def forward(self, panels=None, text=None, embeddings=None):
        if embeddings is None:
            return self.forward_encoder(panels, text)
        else:
            return self.forward_decoder(embeddings)