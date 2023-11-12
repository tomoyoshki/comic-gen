import logging
import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np


from utils.input.input_utils import process_text

def eval_metrics(args, predictions, all_labels):

    if args.stage in {"encode"}:
        mse_loss = nn.MSELoss()
    
        # predictions: predcited embeddings
        # all labels: gt embeddings
    
        # eval cosine similarity 
        cos_sim = None
    
        # eval mse: nn.
        mse = None
        
        return cos_sim, mse
    else:
        pass

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
            
            all_predicted_embeddings.append(embeddings.cpu().numpy())
            all_gt_embeddings.append(gt_embeddings.cpu().numpy())
            if args.stage in {"generate"}:
                all_gt_texts.append(texts[:, -1])
                all_decoded_texts.append(decoded_texts)

    all_predictions = np.concatenate(all_predicted_embeddings, axis=0)
    all_gt = np.concatenate(all_gt_embeddings, axis=0)
        
    # compute metrics
    mean_loss = np.mean(loss_list)
    metrics = eval_metrics(args, all_predictions, all_gt)

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
        #TODO: Add metrics here
        logging.info(f"Test loss: {test_loss: .5f}")

    elif args.stage in {"decode"}:
        logging.info(f"Val loss: {val_loss: .5f}")
        logging.info(f"Val decoding metric1: {val_metrics[0]: .5f}, Val decoding metric2: {val_metrics[1]: .5f}")
        logging.info(f"Test loss: {test_loss: .5f}")
        logging.info(f"Test decoding metric1: {test_metrics[0]: .5f}, Test decoding metric2: {test_metrics[1]: .5f}")

    return val_metrics, val_loss
