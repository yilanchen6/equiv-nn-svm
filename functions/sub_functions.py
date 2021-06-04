import numpy as np
import torch
import random
import os


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def set_deterministic(seed=0):
    """Try to configure the system for reproducible results."""
    if seed is None:
        seed = 0
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
