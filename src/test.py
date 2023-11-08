import warnings

from train_utils.evaluations import eval_pretrained_model

warnings.simplefilter("ignore", UserWarning)

import torch.nn as nn

from data.multi_modal_dataloader import create_dataloader
from train_utils.train_engine import eval_model
from train_utils.selection import select_model, select_loss_func
from utils.params.params import parse_params
from utils.general.load_weight import load_model_weight



def test(args):
    print("= Loading dataloaders...")
    test_dataloader = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)

    print("= Building model and loss function...")
    model = select_model(args)
    loss_func = select_loss_func(args)

    model = load_model_weight(args, model)
    args.model = model

    test_classifier_loss, test_metrics = eval_pretrained_model(args, model, test_dataloader, loss_func)
    print(f"Test classifier loss: {test_classifier_loss: .5f}")
    print(f"Test metrics: {test_metrics}")

    return test_classifier_loss, test_metrics[0], test_metrics[1]


def main_test():
    args = parse_params(mode="test")
    test(args)

if __name__ == "__main__":
    main_test()