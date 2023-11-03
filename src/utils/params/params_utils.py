import os
import getpass

import torch
import yaml

from utils.params.params_path import set_model_weight_folder

def str_to_bool(flag):
    return True if flag.lower().strip() == "true" else False

def get_username():
    return getpass.getuser()

def select_device(device="", batch_size=0, newline=True):
    # CUDA had some problem with GPU selection, therefore this function add envion variable to force GPU selection

    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f"Torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)"  # bytes to MB
        arg = f"cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU"
        arg = "cpu"

    if not newline:
        s = s.rstrip()

    return torch.device(arg)

def set_auto_params(args):
    
    args.device = select_device(args.gpu)
    # set any bool args to bool here
    args.verbose = str_to_bool(args.verbose)
    
    dataset_yaml = f"./data/config.yaml"
    with open(dataset_yaml, "r", errors="ignore") as stream:
        yaml_data = yaml.safe_load(stream)

    args.dataset_config = yaml_data
    args.workers = 4

    # set model path files
    args = set_model_weight_folder(args)
    
    return args