from utils.params.base_params import parse_base_params
from utils.params.params_utils import set_auto_params

def parse_params():
    args = parse_base_params()
    args = set_auto_params(args)
    pass

