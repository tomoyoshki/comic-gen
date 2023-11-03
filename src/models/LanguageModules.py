
import torch
import torch.nn as nn


class LanguageEncoder(nn.Module):
    def __init__(self, args):
        super(LanguageEncoder, self).__init__()

        self.args = args
        self.config = args.dataset_config["LanguageEncoder"]
        
        # raise NotImplementedError("Language Encoder not implemented yet.\n\tPlease implement it in src/models/LanguageModules.py\n\tPlease refer to tokenizer and pretrained language model for reference(e.g. openai)")

    def forward(self):        
        raise NotImplementedError("LanguageEncoder forward pass not implemented yet.")
        pass