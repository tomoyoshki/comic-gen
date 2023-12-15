import torch
import torch.nn as nn

from models.VisionModules import VisionEncoder
from models.LanguageModules import LanguageEncoder
from models.DecoderModules import Decoder
import torch.nn.functional as F

class Codigen(nn.Module):
    
    def __init__(self, args, tokenizer):
        super(Codigen, self).__init__()
        self.args = args
        self.config = args.dataset_config
        
        self.place_holder = nn.Linear(1, 1)
        
        self.vis_encoder = VisionEncoder(args)
        self.lan_encoder = LanguageEncoder(args)
        # 154368
        self.sequential_network = nn.RNN(input_size=1536, hidden_size=768, num_layers=1, batch_first=True)
        
        self.fuse_network = nn.Sequential(
            nn.Linear(768, 1000),
            nn.ReLU(),
            nn.Linear(1000, 768),
        )
        
        self.__init__encoder()

        # self.decoder = None # GPT?
        self.tokenizer = tokenizer
        self.decoder = Decoder(self.args, self.tokenizer)

    # THIS IS DIRECTLY COPIED FROM BaselineModules.py
    def __init__encoder(self):
        self.ground_truth_encoder = LanguageEncoder(self.args)
        for param in self.ground_truth_encoder.parameters():
            param.requires_grad = False  # not update by gradient
            
        fused_dim = (self.args.dataset_config["seq_len"] - 1) * self.args.dataset_config["text_embed_dim"]
        self.fusion_module = nn.Linear(
            fused_dim, self.args.dataset_config["text_embed_dim"]
        )

    def forward_encoder(self, panels, texts):
        """
        Vision input: [b, seq_len, 3, 256, 256] 
        Vision output: [b, seq_len, embedding dimension]

        Text: [b, seq_len, 2 (text token ids, masks[mask no need, need to pass in separately], token length, token dim)
        - Input to language encoder 
            - text token id: [b, seq_len, token len, token dim] (from third dimension of text)
            - mask: [b, seq_len, token len, dim] (from third dimension of text)

        - Output from the langauge encoder
            - text embeddings: [batch_size, seq_len, token_len, embedding dimension]

        - Because we have tokens, unlike images, there is dimension mismatch, and we need to use pool function
        - output from pool
            - text embeddings: [batch_size, seq_len, embedding dimension]
        """
        # encoder
        panel_embeddings = self.vis_encoder(panels) # output: [batch_size, seq_len, embedding_dim] embedding_dim = 768
        
        text_seq_embeddings = []
        # For each panel_text in the sequence until the second last one
        for i in range(self.config["seq_len"] - 1):
            text_token_id = texts[:, i, 0, :]
            text_attention_mask = texts[:, i, 1, :]
            text_embedding = self.lan_encoder(text_token_id, text_attention_mask) # [batch_size, 1, tolen_len, embeddings]
            text_seq_embeddings.append(text_embedding)

        text_seq_embeddings = torch.stack(text_seq_embeddings, dim=1) # [batch_size, seq_len - 1, token_len, embedding_dim]
        pooled_text_embeddings = self.pool(text_seq_embeddings)

        # text_token_id = texts[:, :, 0, :]
        # text_attention_mask = texts[:, :, 1, :]
        # text_embeddings = self.lan_encoder(text_token_id, text_attention_mask) # [batch_size, seq_len, token_len, embedding_dim] 

        # Pool on the token_len dim
        # text_embeddings = self.pool(text_seq_embeddings) # [b, seq_len - 1, emb_dim]
        
        # unsqueeze and expand panel embeddings
        panel_token_embeddings = panel_embeddings.unsqueeze(2).expand(-1, -1, text_seq_embeddings.shape[2], -1) # output: [batch_size, seq_len, token_len, embedding_dim]
        # print(panel_embeddings.shape)

        # normalize the embeddings
        panel_token_embeddings = F.normalize(panel_token_embeddings)
        text_seq_embeddings = F.normalize(text_seq_embeddings)
        panel_embeddings = F.normalize(panel_embeddings)
        pooled_text_embeddings = F.normalize(pooled_text_embeddings)
        
        # concat panel embeddings and text embeddings
        concat_embedding = torch.cat((panel_embeddings[:, :-1], pooled_text_embeddings), dim=-1) # output: [batch_size, seq_len - 1, token_len, embedding_dim * 2]
        concat_token_embedding = torch.cat((panel_token_embeddings[:, :-1, :], text_seq_embeddings), dim=-1)
        # print(concat_embedding.shape)
        
        # reshape concat embedding to feed into sequential network
        sequential_input = concat_embedding.view(concat_embedding.shape[0], concat_embedding.shape[1], -1)
        # print(sequential_input.shape)
        
        # sequential network -> [batch_size, seq_len - 1, token_len * embedding_dim]
        sequential_embedding, _ = self.sequential_network(sequential_input)
        
        b, s, t, dim = concat_token_embedding.shape
        concat_token_embedding = concat_token_embedding.reshape(b, s * t, dim)
        sequential_token_embedding, _ = self.sequential_network(concat_token_embedding)
        sequential_token_embedding = sequential_token_embedding.reshape(b, s, t, dim // 2)
        # print(sequential_embedding.shape)
        
        # sequential embedding -> [batch_size, seq_len - 1, token_len, embedding_dim]
        # sequential_embedding = sequential_embedding.view(sequential_embedding.shape[0], sequential_embedding.shape[1], text_seq_embeddings.shape[2], -1)
        # print(sequential_embedding.shape)
        
        # sequential_embedding -> [batch_size, seq_len, token_len, embedding_dim]
        sequential_embedding = torch.concat([sequential_embedding, panel_embeddings[:, -1, :].unsqueeze(1)], dim=1)
        sequential_token_embedding = torch.concat([sequential_token_embedding, panel_token_embeddings[:, -1, :].unsqueeze(1)], dim=1)
        # print(sequential_embedding.shape)
        
        # What we want: [b, seq_len, embed_dim] -> [b, embed_dim]
        sequential_embedding = torch.mean(sequential_embedding, dim=1)
        sequential_token_embedding = torch.mean(sequential_token_embedding, dim=1)
        # print(sequential_embedding.shape)

        # [b, seq_len, embed_dim] -> [b, seq_len, 1000] -> [b, seq_len, embed_dim]
        fused_embeddings = self.fuse_network(sequential_embedding)
        fused_token_embeddings = self.fuse_network(sequential_token_embedding)
        # print(fused_embeddings.shape)

        # Ground Truth Embeddings
        ground_truth_id = texts[:, -1, 0, :]
        ground_truth_mask = texts[:, -1, 1, :]
        gt_token_embeddings = self.ground_truth_encoder(ground_truth_id, ground_truth_mask)
        gt_embeddings = self.gt_pool(gt_token_embeddings)
        # print(gt_embeddings.shape)

        return fused_embeddings, gt_embeddings, gt_token_embeddings, sequential_token_embedding


    # TODO: TO BE IMPLEMENTED
    def forward_decoder(self, embeddings, gt_token_id=None):
        loss, decoded_text_list = self.decoder(embeddings, gt_token_id)
        return loss, decoded_text_list
    

    def forward(self, panels=None, text=None, embeddings=None):
        """Encode only """
        embeddings, gt_embedding, gt_token_embeddings, sequential_token_embeddings = self.forward_encoder(panels, text)
        loss = None
        decoded_text_list = None

        # if self.args.stage in {"decode"}:
        loss, decoded_text_list = self.forward_decoder(sequential_token_embeddings, text[:, -1, 0, :])
        return embeddings, gt_embedding, loss, decoded_text_list


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
        """Pool over seq len
        Output: [b, seq_len, embedding]
        """
        embedding_transposed = embedding.transpose(1, 2) # [b, embedding, seq_len]
        # Apply max pooling over the token length dimension
        max_pooled = F.max_pool1d(embedding_transposed, kernel_size=embedding_transposed.size(-1))
        max_pooled = max_pooled.squeeze(-1)
        return max_pooled
