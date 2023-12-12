import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class LanguageEncoder(nn.Module):
    def __init__(self, args):
        super(LanguageEncoder, self).__init__()

        self.args = args
        self.dataset_config = args.dataset_config
        self.config = args.dataset_config["LanguageEncoder"]
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True, output_attentions=False)
        # self.positional_embedding = nn.Parameter(torch.empty(1, self.args.dataset_config["text_embed_dim"]))

    def forward(self, text_token_id, text_attention_mask=None):
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        causal_lm_output = self.model(input_ids=text_token_id, attention_mask=text_attention_mask)
        hidden_states = causal_lm_output.hidden_states
        embedding = hidden_states[-1]
        return embedding

class LanguageDecoder(nn.Module):
    def __init__(self, args):
        super(LanguageDecoder, self).__init__()

        self.args = args
        self.config = args.dataset_config["LanguageDecoder"]

        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.max_length = self.args.dataset_config["max_seq_len"]
        self.tokenizer = args.tokenizer

        self.eos_token_id = self.tokenizer.eos_token_id

    def forward(self, embeddings, gt_token_id=None):
        # return self.model(inputs_embeds=embeddings, max_length=self.max_length)
        if self.args.stage in {"encode", "decode"}:
            causal_lm_output = self.model(inputs_embeds=embeddings, labels=gt_token_id)
            loss = causal_lm_output.loss
            # return loss
            
            logits = self.model.lm_head(embeddings)
            predicted_token_ids = logits.argmax(-1)
            
            decoded_text_list = []
            for predicted_token_id in predicted_token_ids:
                decoded_text_list.append(self.tokenizer.decode(predicted_token_id))
            
            actual_text_list = []
            for gti in gt_token_id:
                actual_text_list.append(self.tokenizer.decode(gti))
            # print(f"\n\n\nPredicted: {decoded_text_list}")
            # print(f"Gt token: {gt_token_id.shape}")
            # print(f"Actual: {actual_text_list}\n\n\n")
            return loss, decoded_text_list
        else:
            tokens = self.model.generate(inputs_embeds=embeddings, max_length=self.max_length)
            return tokens