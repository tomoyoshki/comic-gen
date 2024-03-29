import logging
from transformers import GPT2Tokenizer

from models.CodigenModules import (
    Codigen
)

from models.BaselineModules import (
    BaselineLanguageNonSequential,
    BaselineVisionNonSequential,
    BaselineLanguageVisionNonSequential,
    BaselineLanguageSequential,
    BaselineVisionSequential
)

from models.loss import (
    GeneralLoss
)

def select_model(args):
    logging.info("= Loading pretrained GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    args.tokenizer = tokenizer
    
    if args.framework == "Codigen":
        model = Codigen(args, tokenizer) #, tokenizer
    elif args.framework == "Baseline":
        if args.baseline == "LanguageNonSequential":
            model = BaselineLanguageNonSequential(args, tokenizer)
        elif args.baseline == "VisionNonSequential":
            model = BaselineVisionNonSequential(args, tokenizer)
        elif args.baseline == "LanguageVisionNonSequential":
            model = BaselineLanguageVisionNonSequential(args, tokenizer)
        elif args.baseline == "LanguageSequential":
            model = BaselineLanguageSequential(args, tokenizer)
        elif args.baseline == "VisionSequential":
            model = BaselineVisionSequential(args, tokenizer)
        else:
            raise Exception(f"Invalid baseline framework provided: {args.baseline}")
    
    model = model.to(args.device)
    return model

def select_loss_func(args):
    """Initialize the loss function according to the config."""
    if args.framework in {"Codigen", "Baseline"}:
        loss_func = GeneralLoss(args)
    else:
        raise Exception(f"Invalid framework provided: {args.framework}")

    loss_func = loss_func.to(args.device)
    return loss_func