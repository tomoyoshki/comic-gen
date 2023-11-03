from utils.params.base_params import parse_base_params
from utils.params.params_utils import set_auto_params

def parse_params(mode="train"):
    args = parse_base_params()
    args.mode=mode
    args = set_auto_params(args)
    return args

