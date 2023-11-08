import warnings
warnings.simplefilter("ignore", UserWarning)
import sys

import logging
import torch
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from data.multi_modal_dataloader import create_dataloader
from train_utils.train_engine import pretrain
from train_utils.selection import select_model, select_loss_func

from utils.params.params import parse_params


np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)



def train(args):

    logging.info("= Loading dataloaders...")
    train_dataloader = create_dataloader("train", args, batch_size=args.batch_size, workers=args.workers)
    val_dataloader = create_dataloader("val", args, batch_size=args.batch_size, workers=args.workers)
    test_dataloader = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)
    num_batches = len(train_dataloader)
    
    logging.info("= Building model and loss function...")
    model = select_model(args)
    loss_func = select_loss_func(args)
    
    logging.info("= Start pretraining...")
    pretrain(args, model, train_dataloader, val_dataloader, test_dataloader, loss_func, num_batches)

def main_train():
    """The main function of training"""
    args = parse_params()
    train(args)


if __name__ == "__main__":
    main_train()
