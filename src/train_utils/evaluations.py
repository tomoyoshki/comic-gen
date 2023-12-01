import logging
import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


from utils.input.input_utils import process_text

def eval_metrics(args, predictions, all_labels):
    mse_loss = nn.MSELoss(reduction="mean")
    sim_loss = nn.CosineEmbeddingLoss(reduction="mean")
    if args.stage in {"encode"}:
        target = torch.ones(all_labels.shape[0], device=args.device)
        cos_sim = sim_loss(predictions, all_labels, target)
        mse = mse_loss(predictions, all_labels)
        return (cos_sim, mse)

def eval_decoder_metrics(args, gt_texts, decoded_texts, metric="bleu"):
    # def text_eva(pred, gt, metric="bleu"):
    loss = 0
    for i in range(len(gt_texts)):
        pred = decoded_texts[i]
        pred = [p.split() for p in pred]
        gt = gt_texts[i]
        gt = [g.split() for g in gt]
        if metric == "bleu":
            for j in range(len(pred)):
                bleu = sentence_bleu(pred[j], gt[j])
                loss += bleu
    return loss
        # elif metric == "spacy":
            # return pred.similarity(gt)
        # else:
            # return "not implemented"

def eval_pretrained_model(args, model, dataloader, loss_func):
    labels = []
    loss_list = []
    all_gt_embeddings = []
    all_predicted_embeddings = []
    all_gt_texts = []
    all_decoded_texts = []

    with torch.no_grad():
        for panels, texts in tqdm(dataloader, total=len(dataloader)):

            """Eval pretrain loss."""
            tokens = process_text(args, texts)
            
            panels = panels.to(args.device)
            tokens = tokens.to(args.device)
            
            embeddings, gt_embeddings, decoded_tokens, decoded_texts = model(panels, tokens)
            
            if args.stage in {"encode"}:
                loss = loss_func(embeddings, gt_embeddings)
            elif args.stage in {"decode"}:
                loss = decoded_tokens
            else:
                loss = torch.tensor(0.0, device=args.device)

            loss_list.append(loss.item())
            
            all_predicted_embeddings.append(embeddings)
            all_gt_embeddings.append(gt_embeddings)
            if args.stage in {"decode", "generate"}:
                all_gt_texts.append(texts[:, -1])
                all_decoded_texts.append(decoded_texts)

    all_predictions = torch.concatenate(all_predicted_embeddings, axis=0)
    all_gt = torch.concatenate(all_gt_embeddings, axis=0)
        
    # compute metrics
    mean_loss = np.mean(loss_list)
    if args.stage in {"encode"}:
        metrics = eval_metrics(args, all_predictions, all_gt)
    else:
        metrics = eval_decoder_metrics(args, all_gt_texts, all_decoded_texts)

    return mean_loss, metrics

def eval_model(args, epoch, model, val_dataloader, test_dataloader, loss_func, train_loss=None):
    
    if args.mode == "train":
        logging.info(f"Training {args.stage} loss: {train_loss: .5f} \n")


    # loss is general loss same as the pretrain, see GeneralLoss in loss.py
    # metrics is calculated in eval_metrics, in encoding stage it is [cos_sim, mse], decoding state yet implemented
    val_loss, val_metrics = eval_pretrained_model(args, model, val_dataloader, loss_func)
    test_loss, test_metrics = eval_pretrained_model(args, model, test_dataloader, loss_func)


    if args.stage in {"encode"}:    
        logging.info(f"Val loss: {val_loss: .5f}")
        
        # print("val metrics", type(val_metrics[0]), type(val_metrics[1]))
        # print(val_metrics[0].shape, val_metrics[1].shape)
        logging.info(f"Val cosine similarity: {val_metrics[0]: .5f}, Val MSE: {val_metrics[1]: .5f}")
        logging.info(f"Test loss: {test_loss: .5f}")
        logging.info(f"Test cosine similarity: {test_metrics[0]: .5f}, Test MSE: {test_metrics[1]: .5f}")

    elif args.stage in {"decode"}:
        logging.info(f"Val loss: {val_loss: .5f}")
        logging.info(f"Val BLEU Score: {val_metrics: .5f}, Val decoding metric2: {0: .5f}")
        logging.info(f"Test loss: {test_loss: .5f}")
        logging.info(f"Test BLEU Score: {test_metrics: .5f}, Test decoding metric2: {0: .5f}")

    return val_metrics, val_loss
