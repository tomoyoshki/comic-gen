from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LanguageModules import LanguageEncoder
from models.DecoderModules import Decoder
from models.VisionModules import VisionEncoder

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
    

class BaselineVisionNonSequential(BaseModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.__init__encoder()

    def __init__encoder(self):
        self.vision_encoder = VisionEncoder(self.args)
        self.ground_truth_encoder = LanguageEncoder(self.args)
        for param in self.ground_truth_encoder.parameters():
            param.requires_grad = False  # not update by gradient
            
        fused_dim = (self.args.dataset_config["seq_len"]) * self.args.dataset_config["text_embed_dim"]
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
        panel_embeddings = self.vision_encoder(panels)
        panel_embeddings = panel_embeddings.reshape(panel_embeddings.shape[0], -1)
        panel_embeddings = self.fusion_module(panel_embeddings)
        ground_truth_id = texts[:, -1, 0, :]
        ground_truth_mask = texts[:, -1, 1, :]
        gt_seq_embeddings = self.ground_truth_encoder(ground_truth_id, ground_truth_mask)
        gt_embeddings = self.pool(gt_seq_embeddings)
        return panel_embeddings, gt_embeddings, None
    



class BaselineLanguageVisionNonSequential(BaseModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.__init__encoder()

    def __init__encoder(self):
        self.vision_encoder = VisionEncoder(self.args)
        self.language_encoder = LanguageEncoder(self.args)
        self.ground_truth_encoder = LanguageEncoder(self.args)
        for param in self.ground_truth_encoder.parameters():
            param.requires_grad = False  # not update by gradient
            
        self.fusion_module = nn.Linear(
            2 * self.args.dataset_config["text_embed_dim"], self.args.dataset_config["text_embed_dim"]
        )
        

    def pool(self, embedding):
        """Pool over token length
        Codigen Pool Input: [b, seq_len, token_len, embedding]
        Output: [b, seq_len, embedding]
        """
        # Reshape and transpose: [batch_size * seq_len, embedding_dim, token_len]
        b, seq_len, token_len, embedding_dim = embedding.shape
        embedding_reshaped = torch.permute(embedding, (0, 1, 3, 2))
        embedding_reshaped = embedding_reshaped.reshape(b * seq_len, embedding_dim, token_len)
        # embedding_reshaped = embedding.reshape(b * seq_len, embedding_dim, token_len)

        # Apply max pooling over the token length dimension
        max_pooled = F.max_pool1d(embedding_reshaped, kernel_size=token_len)
        max_pooled = max_pooled.squeeze(-1)
        # Reshape back to [batch_size, seq_len, embedding_dim]
        max_pooled = max_pooled.reshape(b, seq_len, embedding_dim)

        return max_pooled


    def gt_pool(self, embedding):
        embedding_transposed = embedding.transpose(1, 2) # [b, embedding, seq_len]
        # Apply max pooling over the token length dimension
        max_pooled = F.max_pool1d(embedding_transposed, kernel_size=embedding_transposed.size(-1))
        max_pooled = max_pooled.squeeze(-1)
        return max_pooled
    

    
    def forward_encoder(self, panels, texts):
        token_text_embeddings = []
        # For each panel_text in the sequence until the second last one
        for i in range(self.dataset_config["seq_len"] - 1):
            text_token_id = texts[:, i, 0, :]
            text_attention_mask = texts[:, i, 1, :]
            text_embedding = self.language_encoder(text_token_id, text_attention_mask)
            token_text_embeddings.append(text_embedding)
        token_text_embeddings = torch.stack(token_text_embeddings, dim=1) # [b, seq_len-1, token_len, embed_dim]
        text_embeddings = self.pool(token_text_embeddings) # [b, seq_len - 1, embed_dim]

        panel_embeddings = self.vision_encoder(panels) # output: [batch_size, seq_len, embedding_dim] embedding_dim = 768
        
        
        mean_language = torch.mean(text_embeddings, dim=1)  
        mean_vision = torch.mean(panel_embeddings, dim=1)  
        concatenated_embedddings = torch.cat((mean_language, mean_vision), dim=1)
        concatenated_embedddings = self.fusion_module(concatenated_embedddings)

        ground_truth_id = texts[:, -1, 0, :]
        ground_truth_mask = texts[:, -1, 1, :]
        gt_seq_embeddings = self.ground_truth_encoder(ground_truth_id, ground_truth_mask)
        gt_embeddings = self.gt_pool(gt_seq_embeddings)

        return concatenated_embedddings, gt_embeddings, None


class BaselineVisionSequential(BaseModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.__init__encoder()

    def __init__encoder(self):
        self.vision_encoder = VisionEncoder(self.args)
        self.ground_truth_encoder = LanguageEncoder(self.args)
        for param in self.ground_truth_encoder.parameters():
            param.requires_grad = False  # not update by gradient
            
        fused_dim = (self.args.dataset_config["seq_len"]) * self.args.dataset_config["text_embed_dim"]
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
        panel_embeddings = self.vision_encoder(panels)
        panel_embeddings = panel_embeddings.reshape(panel_embeddings.shape[0], -1)
        panel_embeddings = self.fusion_module(panel_embeddings)
        ground_truth_id = texts[:, -1, 0, :]
        ground_truth_mask = texts[:, -1, 1, :]
        gt_seq_embeddings = self.ground_truth_encoder(ground_truth_id, ground_truth_mask)
        gt_embeddings = self.pool(gt_seq_embeddings)
        return panel_embeddings, gt_embeddings, None

      
class BaselineLanguageSequential(BaseModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        
        self.__init__encoder()

    def __init__encoder(self):
        self.language_encoder = LanguageEncoder(self.args)
        self.ground_truth_encoder = LanguageEncoder(self.args)
        for param in self.ground_truth_encoder.parameters():
            param.requires_grad = False  # not update by gradient
            
        fused_dim = (self.args.dataset_config["seq_len"] - 1) * self.args.dataset_config["text_embed_dim"]
        
        self.sequential_network = nn.RNN(input_size=self.args.dataset_config["text_embed_dim"], hidden_size=self.args.dataset_config["text_embed_dim"], num_layers=1, batch_first=True)
        self.fusion_module = nn.Linear(
            self.args.dataset_config["text_embed_dim"], self.args.dataset_config["text_embed_dim"]
        )
        

    def pool(self, embedding):
        """Pool over token length
        Codigen Pool Input: [b, seq_len, token_len, embedding]
        Output: [b, seq_len, embedding]
        """
        # Reshape and transpose: [batch_size * seq_len, embedding_dim, token_len]
        b, seq_len, token_len, embedding_dim = embedding.shape
        embedding_reshaped = torch.permute(embedding, (0, 1, 3, 2))
        embedding_reshaped = embedding_reshaped.reshape(b * seq_len, embedding_dim, token_len)
        # embedding_reshaped = embedding.reshape(b * seq_len, embedding_dim, token_len)

        # Apply max pooling over the token length dimension
        max_pooled = F.max_pool1d(embedding_reshaped, kernel_size=token_len)
        max_pooled = max_pooled.squeeze(-1)
        # Reshape back to [batch_size, seq_len, embedding_dim]
        max_pooled = max_pooled.reshape(b, seq_len, embedding_dim)

        return max_pooled

    def gt_pool(self, embedding):
        embedding_transposed = embedding.transpose(1, 2)
        # Apply max pooling over the sequence length dimension
        max_pooled = F.max_pool1d(embedding_transposed, kernel_size=embedding_transposed.size(-1))
        max_pooled = max_pooled.squeeze(-1)
        return max_pooled
    
    def forward_encoder(self, panels, texts):

        token_text_embeddings = []

        # For each panel_text in the sequence until the second last one
        for i in range(self.dataset_config["seq_len"] - 1):
            text_token_id = texts[:, i, 0, :]
            text_attention_mask = texts[:, i, 1, :]
            text_embedding = self.language_encoder(text_token_id, text_attention_mask)

            token_text_embeddings.append(text_embedding)

        token_text_embeddings = torch.stack(token_text_embeddings, dim=1) # [b, seq_len-1, token_len, embed_dim]
        text_embeddings = self.pool(token_text_embeddings) # [b, seq_len - 1, embed_dim]
        seq_text_embeddings, _ = self.sequential_network(text_embeddings) # [b, seq_len - 1, embed_dim]
        seq_text_embeddings = seq_text_embeddings.mean(dim=1) # [b, embed_dim]
        seq_text_embeddings = self.fusion_module(seq_text_embeddings) # [b, embed_dim]
    
        ground_truth_id = texts[:, -1, 0, :]
        ground_truth_mask = texts[:, -1, 1, :]
        token_gt_embeddings = self.ground_truth_encoder(ground_truth_id, ground_truth_mask)
        gt_embeddings = self.gt_pool(token_gt_embeddings)

        return seq_text_embeddings, gt_embeddings, token_text_embeddings