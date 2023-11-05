import torch.nn as nn
from models.LanguageModules import LanguageDecoder

class Decoder(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.backbone = LanguageDecoder(self.args)
        self.tokenizer = tokenizer

    def forward(self, embedding):
        tokens = self.backbone(embedding)

        texts = None
        if self.args.stage in {"generate"}:
            texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in tokens]

        return tokens, texts