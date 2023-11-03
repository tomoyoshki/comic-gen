import logging
import torch
from tqdm import tqdm
import numpy as np

def eval_metrics(args, predictions, all_labels):
    pass

def eval_pretrained_model(args, model, dataloader, loss_func):
    labels = []
    loss_list = []
    all_gt_embeddings = []
    all_predicted_embeddings = []
    
    with torch.no_grad():
        for panel, text in tqdm(dataloader, total=len(dataloader)):
            label = label.argmax(dim=1, keepdim=False) if label.dim() > 1 else label
            labels.append(label.cpu().numpy())

            """Eval pretrain loss."""
            predicted_embedding, gt_embedding = model(panel, text)
            loss = loss_func(predicted_embedding, gt_embedding)
            loss_list.append(loss)
            
            all_predicted_embeddings.append(predicted_embedding.cpu().numpy())
            all_gt_embeddings.append(gt_embedding.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # compute metrics
    mean_loss = np.mean(loss_list)
    metrics = eval_metrics(args, all_predictions, all_labels)

    return mean_loss, metrics

def eval_model(args, epoch, model, val_dataloader, test_dataloader, loss_func, train_loss):
    logging.info(f"Training loss: {train_loss: .5f} \n")


    val_loss, val_metrics = eval_pretrained_model(args, model, val_dataloader, loss_func)
    test_loss, test_metrics = eval_pretrained_model(args, model, test_dataloader, loss_func)


    if args.stage in {"encode"}:    
        logging.info(f"Val loss: {val_loss: .5f}")
        logging.info(f"Test loss: {test_loss: .5f}")
    else:
        """TODO: add decoding metric here"""
        logging.info(f"Val loss: {val_loss: .5f}")
        logging.info(f"Val decoding metric1: {val_metrics[0]: .5f}, Val decoding metric2: {val_metrics[1]: .5f}")
        logging.info(f"Test loss: {test_loss: .5f}")
        logging.info(f"Test decoding metric1: {test_metrics[0]: .5f}, Test decoding metric2: {test_metrics[1]: .5f}")

    return val_metrics, val_loss
