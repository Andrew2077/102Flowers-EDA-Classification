import json

import numpy as np
import torch


def set_global_seed(SEED: int)->None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.set_num_threads(1)
    # torch.autograd.set_detect_anomaly(True)
    
def accuray_fn(ypred, y_true):
    """
    Calculate the accuracy of the prediction
    """
    return (ypred.argmax(dim=1) == y_true).sum() / len(ypred)


def load_configs(path):
    return json.load(open(path, "r"))