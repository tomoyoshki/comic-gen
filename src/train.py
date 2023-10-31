import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import sys

import logging
import torch
import numpy as np

from utils.params.params import parse_params


np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)



def train(args):
    print("Training...")
    pass

def main_train():
    """The main function of training"""
    args = parse_params()
    train(args)


if __name__ == "__main__":
    main_train()
