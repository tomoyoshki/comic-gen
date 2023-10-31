from torch import optim as optim

def define_optimizer(args, parameters):
    """Define the optimizer."""
    optimizer_config = args.dataset_config[args.learn_framework]["pretrain_optimizer"]
    optimizer_name = optimizer_config["name"]
    weight_decay = optimizer_config["weight_decay"]

    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            parameters,
            lr=optimizer_config["start_lr"],
            weight_decay=weight_decay,
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            parameters,
            lr=optimizer_config["start_lr"],
            weight_decay=weight_decay,
        )
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            parameters,
            lr=optimizer_config["start_lr"],
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented.")

    return optimizer
