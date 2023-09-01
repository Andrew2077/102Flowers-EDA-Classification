import json
import os
import tarfile
import warnings

import numpy as np
import pandas as pd
import requests
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchsummary
import torchvision
from engine.data_download import download, download_extrac_all, extract_tgz
from engine.data_processing import FlowerDataset, prepare_df
from engine.models import Resnet50Flower102
from engine.train import training_loop, training_step
from engine.utils import accuray_fn, load_configs, set_global_seed
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

configs = load_configs(r"config.json")
env = configs["config"]["env"]
if env == "notebook":
    # from tqdm import tqdm_notebook as tqdm
    from tqdm.notebook import tnrange as tqdm

    ncols = 100
elif env == "local":
    from tqdm import tqdm

    ncols = None
else:
    raise ValueError("env must be either notebook or local")

# warnings.filterwarnings("ignore")


dataset_url = configs["urls"]["data_url"]
labels = configs["urls"]["labels_url"]
splits = configs["urls"]["split_url"]

data_root = configs["dir"]["data_path"]
labels_Path = configs["dir"]["labels_path"]
split_path = configs["dir"]["split_path"]

overwirte_data = configs["config"]["overwrite_data"]
tzg_path = configs["dir"]["tzg_path"]

download_extrac_all(dataset_url, labels, splits, tzg_path, OVERWRITE=overwirte_data)


if __name__ == "__main__":
    # * Gloabl hyperparameters
    SEED = configs["config"]["seed"]
    DEVICE = configs["config"]["device"]
    PRETRAINED_WEIGHTS = configs["config"]["pretrained_weights"]
    FREEZE_RESNET = configs["config"]["freeze_resnet"]
    models_path = configs["dir"]["model_path"]
    set_global_seed(SEED)

    # * Training hyperparameters
    LEARNING_RATE = configs["config"]["learning_rate"]
    NUM_EPOCHS = configs["config"]["epochs"]
    TRAIN_BATCH_SIZE = configs["config"]["train_batch_size"]
    VALIDATION_BATCH_SIZE = configs["config"]["val_batch_size"]
    SCHUFFLE_TRAIN = configs["config"]["schuffle_train"]
    SCHUFFLE_VALIDATION = configs["config"]["schuffle_val"]

    transformsations = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_split, test_split, val_split = prepare_df(split_path, labels_Path, data_root)

    train_dataset = FlowerDataset(train_split, transform=transformsations)
    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=SCHUFFLE_TRAIN
    )

    val_dataset = FlowerDataset(val_split, transform=transformsations)
    val_loader = DataLoader(
        val_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=SCHUFFLE_VALIDATION
    )

    model = Resnet50Flower102(
        pretrained=PRETRAINED_WEIGHTS,
        freeze_Resnet=FREEZE_RESNET,
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_history = training_loop(
        model,
        loss_fn,
        accuray_fn,
        optimizer,
        train_loader,
        val_loader,
        DEVICE,
        NUM_EPOCHS,
        models_path,
        ncols,
    )
