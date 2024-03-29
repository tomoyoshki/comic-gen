from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler


def define_lr_scheduler(args, optimizer):
    """
    Define the learning rate scheduler
    """
    classifier_config = args.dataset_config[args.framework]
    optimizer_config = args.dataset_config[args.framework]["pretrain_optimizer"]
    scheduler_config = args.dataset_config[args.framework]["pretrain_lr_scheduler"]

    if scheduler_config["name"] == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(scheduler_config["train_epochs"] - scheduler_config["warmup_epochs"])
            if scheduler_config["warmup_prefix"]
            else scheduler_config["train_epochs"],
            cycle_mul=1.0,
            lr_min=optimizer_config["min_lr"],
            warmup_lr_init=optimizer_config["warmup_lr"],
            warmup_t=scheduler_config["warmup_epochs"],
            cycle_limit=1,
            t_in_epochs=True,
            warmup_prefix=scheduler_config["warmup_prefix"],
        )
    elif scheduler_config["name"] == "step":
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=scheduler_config["decay_epochs"],
            decay_rate=scheduler_config["decay_rate"],
            warmup_lr_init=optimizer_config["warmup_lr"],
            warmup_t=scheduler_config["warmup_epochs"],
            t_in_epochs=True,
        )
    else:
        raise Exception(f"Unknown LR scheduler: {classifier_config['lr_scheduler']}")

    return lr_scheduler