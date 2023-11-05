import torch

def process_text(args, texts):
    batch_size = texts.shape[0]
    texts = list(texts.flatten())
    tokens = args.tokenizer(texts, padding='max_length', truncation=True, max_length=args.dataset_config["max_seq_len"], return_tensors="pt")
    input_ids = tokens["input_ids"].reshape(batch_size, args.dataset_config["seq_len"], -1)
    attention_mask = tokens["attention_mask"].reshape(batch_size, args.dataset_config["seq_len"], -1)
    
    # [b, seq_len, dim], [b, seq_len, dim]
    tokens = torch.stack([input_ids, attention_mask], dim=2)
    return tokens