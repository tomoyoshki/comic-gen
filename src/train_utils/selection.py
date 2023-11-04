from models.CodigenModules import (
    Codigen
)

from models.loss import (
    CodigenLoss
)

def select_model(args):
    if args.framework == "Codigen":
        model = Codigen(args)
    elif args.framework == "Baseline":
        # raise Exception(f"Invalid framework provided: {args.framework}")
    
    model = model.to(args.device)
    return model

def select_loss_func(args):
    """Initialize the loss function according to the config."""
    if args.framework == "Codigen":
        loss_func = CodigenLoss(args)
    else:
        raise Exception(f"Invalid train mode provided: {args.train_mode}")

    loss_func = loss_func.to(args.device)
    return loss_func