import os
import time

import logging
import torch
import numpy as np
from tqdm import tqdm
from utils.general.load_weight import load_model_weight

from utils.input.input_utils import process_text
from train_utils.optimizers import define_optimizer
from train_utils.schedulers import define_lr_scheduler
from train_utils.evaluations import eval_model

def pretrain(
    args,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    loss_func,
    num_batches,
):
    """
    The supervised training function for tbe backbone network,
    used in train of supervised mode or fine-tune of foundation models.
    """
    # model config
    classifier_config = args.dataset_config[args.framework]

    # Define the optimizer and learning rate scheduler
    optimizer = define_optimizer(args, model.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)

    # Training loop
    logging.info("---------------------------Start Pretraining Classifier-------------------------------")
    start = time_sync()

    best_val_acc = 0
    best_val_loss = np.inf

    best_weight = os.path.join(args.weight_folder, f"model_weights/{args.framework}_{args.stage}_best.pt")
    latest_weight = os.path.join(args.weight_folder, f"model_weights/{args.framework}_{args.stage}_latest.pt")
    
    if args.stage in {"decode"}:
        model = load_model_weight(args, model)
        for name, param in model.named_parameters():
            if "decoder" not in name:
                param.requires_grad = False
    val_epochs = 5

    for epoch in range(classifier_config["pretrain_lr_scheduler"]["train_epochs"]):
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        model.train()
        args.epoch = epoch

        # training loop
        train_loss_list = []

        # regularization configuration

        for i, (panels, texts) in tqdm(enumerate(train_dataloader), total=num_batches):
            
            tokens = process_text(args, texts)
            panels = panels.to(args.device)
            tokens = tokens.to(args.device)
            embeddings, gt_embeddings, decoded_tokens, decoded_texts = model(panels, tokens)
            
            if args.stage in {"encode"}:
                loss = loss_func(embeddings, gt_embeddings)
            elif args.stage in {"decode"}:
                loss = decoded_tokens
            else:
                raise Exception("Do not run generate stage in pretrain mode")
            # back propagation
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss_list.append(loss.item())
            

        if epoch % 5 == 0 and args.stage in {"decode"}:
            print(f"Decoded: {decoded_texts[0]}")
            print(f"GT: {texts[0]}")

        # validation and logging
        if epoch % val_epochs == 0:
            train_loss = np.mean(train_loss_list)
            val_acc, val_loss = eval_model(
                args,
                epoch,
                model,
                val_dataloader,
                test_dataloader,
                loss_func,
                train_loss,
            )

            # Save the latest model, only the backbone parameters are saved
            torch.save(model.state_dict(), latest_weight)

            # Save the best model according to validation result
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()