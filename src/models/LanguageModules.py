
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


class LanguageEncoder(nn.Module):
    def __init__(self, args):
        super(LanguageEncoder, self).__init__()

        self.args = args
        self.dataset_config = args.dataset_config
        self.config = args.dataset_config["LanguageEncoder"]
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True, output_attentions=False)

    def forward(self, text_token_id, text_attention_mask=None):
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        causal_lm_output = self.model(input_ids=text_token_id, attention_mask=text_attention_mask)
        hidden_states = causal_lm_output.hidden_states
        embedding = hidden_states[-1]
        embedding_transposed = embedding.transpose(1, 2)
        # Apply max pooling over the sequence length dimension
        max_pooled = F.max_pool1d(embedding_transposed, kernel_size=embedding_transposed.size(-1))
        embedding = max_pooled.squeeze(-1)
        return embedding

class LanguageDecoder(nn.Module):
    def __init__(self, args):
        super(LanguageDecoder, self).__init__()

        self.args = args
        self.config = args.dataset_config["LanguageDecoder"]
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, embeddings):
        return self.model.generate(inputs_embeds=embeddings, max_length=100)