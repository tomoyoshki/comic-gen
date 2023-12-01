import torch.nn as nn
from models.LanguageModules import LanguageDecoder

class Decoder(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.backbone = LanguageDecoder(self.args)
        self.tokenizer = tokenizer

    def forward(self, embedding, gt_token_id=None):
        if self.args.stage in {"decode"}:
            loss, decoded_text_list = self.backbone(embedding, gt_token_id)
            return loss, decoded_text_list

        tokens = self.backbone(embedding, gt_token_id)
        if self.args.stage in {"generate"}:
            texts = [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in tokens]

        return tokens, texts