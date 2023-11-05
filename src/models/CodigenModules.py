import torch.nn as nn


class Codigen(nn.Module):
    def __init__(self, args):
        super(Codigen, self).__init__()

        self.args = args
        self.config = args.dataset_config["Codigen"]
        
        self.place_holder = nn.Linear(1, 1)
        
        self.encoder = {
            "images": None,
            "texts": None,
        }
    
        
        self.sequential_network = None

        self.decoder = None

    def forward_encoder(self, panels, texts):
        
        # encoder
        sequential_embedding = None
        for i in range(0, 3):
            panel_embedding = self.encoder["image"](panels[i])
            text_embedding = self.encoder["texts"](texts[i])

            sequential_embedding = self.sequential_network[i](panel_embedding, text_embedding, sequential_embedding)
        
        panel_embedding = self.encoder["image"](panels[-1])
        fused_embeddings = self.sequential_network[-1](panel_embedding, sequential_embedding)
        
        return fused_embeddings

    def forward_decoder(self, embeddings):
        decoded_text = self.decoder(embeddings)
        return decoded_text
    
    def forward(self, panels=None, text=None, embeddings=None):
        if embeddings is None:
            return self.forward_encoder(panels, text)
        else:
            return self.forward_decoder(embeddings)