import torch.nn as nn
from models.DecoderModules import Decoder

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
    
        
        self.sequential_network = nn.Linear(visual_emb_dim + lang_emb_dim + seq_dim, seq_dim)
        self.decoder = Decoder(self.args, self.tokenizer)

    def forward_encoder(self, panels, texts):

        """
        V (b, seq_len, 3, dim, dim) -> 



        RNN = RNN(RNN_output, token)
        previous_input = (V + L + previous_input) fusion(Concat, Transformer (Attention) Fusion, Linear Layer, MLP, LSTM, RNN)
        L

        V P  
        L
        """
        
        # encoder
        sequential_embedding = None
        for i in range(0, 3):
            panel_embedding = self.vision_encoder(panels[i]) # V (v, 1, 3, image_dim, image_dim) -> (b, visual_emb_dim)
            text_embedding = self.language_encoder(texts[i]) # L (b, token_len, emb_dim) -> pool -> (b, lang_emb_dim)
            # check VisBaseline stuff
            

            sequential_embedding = self.sequential_network[i](panel_embedding, text_embedding, sequential_embedding)
        
        panel_embedding = self.encoder["image"](panels[-1])
        fused_embeddings = self.sequential_network[-1](panel_embedding, sequential_embedding)
        
        return fused_embeddings

    def forward_decoder(self, embeddings, gt_token_id=None):
        token, text = self.decoder(embeddings, gt_token_id)
        return token, text
    
    def forward(self, panels=None, text=None, embeddings=None):
        if embeddings is None:
            return self.forward_encoder(panels, text)
        else:
            return self.forward_decoder(embeddings)
