import argparse
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
from engine.experiment import create_writer, set_experiment_params
from engine.models import Resnet50Flower102
from engine.train import training_loop, training_step
from engine.utils import accuray_fn, load_configs, set_global_seed
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter()
global_configs = load_configs(r"config/global-configs.json")
env = global_configs["config"]["env"]
if env == "notebook":
    # from tqdm import tqdm_notebook as tqdm
    from tqdm.notebook import tnrange as tqdm

    ncols = 100
elif env == "local":
    from tqdm import tqdm

    ncols = None
else:
    raise ValueError("env must be either notebook or local")


dataset_url = global_configs["urls"]["data_url"]
labels = global_configs["urls"]["labels_url"]
splits = global_configs["urls"]["split_url"]

data_root = global_configs["dir"]["data_path"]
labels_Path = global_configs["dir"]["labels_path"]
split_path = global_configs["dir"]["split_path"]
tzg_path = global_configs["dir"]["tzg_path"]
model_path = global_configs["dir"]["model_path"]


overwirte_data = global_configs["config"]["overwrite_data"]

download_extrac_all(dataset_url, labels, splits, tzg_path, OVERWRITE=overwirte_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configs")
    parser.add_argument(
        "--config",
        type=str,
        default=r"config/experiment1.json",
        help="path to the experiment config file",
    )
    parser.add_argument(
        "--result_name", type=str, default=None, help="path to the output csv file"
    )
    args = parser.parse_args()

    (
        expriment_name,
        model_name,
        extra,
        PRETRAINED_WEIGHTS,
        FREEZE_RESNET,
        SEED,
        DEVICE,
        TRAIN_BATCH_SIZE,
        VALIDATION_BATCH_SIZE,
        NUM_EPOCHS,
        LEARNING_RATE,
        SCHUFFLE_TRAIN,
        SCHUFFLE_VALIDATION,
        SCHUFFLE_TEST,
    ) = set_experiment_params(args.config)

    set_global_seed(SEED)

    writer = create_writer(expriment_name, model_name, extra)
    transformsations = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # * ImageNet distribution params
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
        model_path,
        model_name,
        ncols,
        writer,
    )

    # * save model history
    if os.path.exists("results") == False:
        os.mkdir("results")

    if args.result_name == None:
        pd.DataFrame(model_history).to_csv(f"results/{expriment_name}.csv")
    else:
        pd.DataFrame(model_history).to_csv(args.result_name)
