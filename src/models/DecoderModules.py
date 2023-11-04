import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, args, model, tokenizer):
        super().__init__()
        self.args = args
        self.backbone = model
        self.tokenizer = tokenizer

    def forward(self, embedding, return_text=False):
        # Generate a sequence of token ids from the embedding
        tokens = self.transformer_model.generate(input_embeds=embedding, max_length=50)
        
        texts = None
        # Only return text during application stage
        # During decoder training, use tokens to calculate loss only
        if return_text:
            texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in tokens]

        return tokens, texts