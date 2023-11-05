from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LanguageModules import LanguageEncoder
from models.DecoderModules import Decoder

class BaseModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseModel, self).__init__()

        self.args = args
        self.dataset_config = args.dataset_config
        self.config = args.dataset_config["Baseline"]
        self.tokenizer = tokenizer
        
        # init encoder, decoder, and sequential_network
        self.__init_encoder__()
        self.__init_decoder__()
    
    @abstractmethod
    def __init_encoder__(self):
        pass
    
    def __init_decoder__(self):
        self.decoder = Decoder(self.args, self.tokenizer)

    @abstractmethod
    def forward_encoder(self, panels, texts):
        pass

    def forward_decoder(self, embeddings, gt_token_id=None):
        token, text = self.decoder(embeddings, gt_token_id)
        return token, text
    
    def forward(self, panels, text):
        embeddings, gt_embedding, seq_embedding = self.forward_encoder(panels, text)

        if self.args.stage in {"encode"}:
            """Encode only """
            return embeddings, gt_embedding, None, None

        decoded_tokens, decoded_texts = self.forward_decoder(seq_embedding, text[:, -1, 0, :])
        return embeddings, gt_embedding, decoded_tokens, decoded_texts


class BaselineLanguageNonSequential(BaseModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        
        self.__init__encoder()

    def __init__encoder(self):
        self.language_encoder = LanguageEncoder(self.args)
        self.ground_truth_encoder = LanguageEncoder(self.args)
        for param in self.ground_truth_encoder.parameters():
            param.requires_grad = False  # not update by gradient
            
        fused_dim = (self.args.dataset_config["seq_len"] - 1) * self.args.dataset_config["text_embed_dim"]
        self.fusion_module = nn.Linear(
            fused_dim, self.args.dataset_config["text_embed_dim"]
        )
        

    def pool(self, embedding):
        embedding_transposed = embedding.transpose(1, 2)
        # Apply max pooling over the sequence length dimension
        max_pooled = F.max_pool1d(embedding_transposed, kernel_size=embedding_transposed.size(-1))
        max_pooled = max_pooled.squeeze(-1)
        return max_pooled

    
    def forward_encoder(self, panels, texts):
        text_seq_embeddings = []

        # For each panel_text in the sequence until the second last one
        for i in range(self.dataset_config["seq_len"] - 1):
            text_token_id = texts[:, i, 0, :]
            text_attention_mask = texts[:, i, 1, :]
            text_embedding = self.language_encoder(text_token_id, text_attention_mask)

            text_seq_embeddings.append(text_embedding)
        
        text_seq_embeddings = torch.concatenate(text_seq_embeddings, dim=-1)
        text_seq_embeddings = self.fusion_module(text_seq_embeddings)

        ground_truth_id = texts[:, -1, 0, :]
        ground_truth_mask = texts[:, -1, 1, :]
        gt_seq_embeddings = self.ground_truth_encoder(ground_truth_id, ground_truth_mask)
        
        text_embeddings = self.pool(text_seq_embeddings)
        gt_embeddings = self.pool(gt_seq_embeddings)

        return text_embeddings, gt_embeddings, text_seq_embeddings