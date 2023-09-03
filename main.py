import argparse
import json
import os
import tarfile
import warnings
from typing import Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import requests
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchsummary
import torchvision
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms

from engine.data_download import download, download_extrac_all, extract_tgz
from engine.data_processing import (
    FlowerDataset,
    prepare_df,
    prepare_splits,
    transformations,
)
from engine.experiment import create_writer, set_experiment_params
from engine.gradcam import GradCAM
from engine.models import Resnet50Flower102
from engine.train import training_loop, training_step
from engine.utils import accuray_fn, load_configs, set_global_seed

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
        SHUFFLE_TRAIN,
        SHUFFLE_VALIDATION,
        SCHUFFLE_TEST,
    ) = set_experiment_params(args.config)

    set_global_seed(SEED)

    custom_writer = create_writer(expriment_name, model_name, extra)

    train_loader, val_loader, test_loader = prepare_splits(
        split_path,
        labels_Path,
        data_root,
        transformations,
        TRAIN_BATCH_SIZE,
        VALIDATION_BATCH_SIZE,
        VALIDATION_BATCH_SIZE,
        SHUFFLE_TRAIN,
        SHUFFLE_VALIDATION,
        SCHUFFLE_TEST,
    )
    test_tensor, test_target = test_loader.dataset[SEED]
    test_tensor, test_target = test_tensor.unsqueeze(0).to(DEVICE), test_target.to(
        DEVICE
    )

    model = Resnet50Flower102(
        device=DEVICE,
        pretrained=PRETRAINED_WEIGHTS,
        freeze_Resnet=FREEZE_RESNET,
    )
    
    with open("config/flower_to_name.json") as f:
        flower_to_name = json.load(f)
    gradcam = GradCAM(model=model, class_dict=flower_to_name)

    CCE = nn.CrossEntropyLoss()
    ADAM = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_history = training_loop(
        model=gradcam.model,
        gradcam=gradcam,
        test_tensor=test_tensor,
        test_target=test_target,
        loss_fn=CCE,
        accuray_fn=accuray_fn,
        optimizer=ADAM,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        models_direcotry=model_path,
        model_name=model_name,
        tqdm_cols=ncols,
        writer=custom_writer,
    )

    # * save model history
    if os.path.exists("results") == False:
        os.mkdir("results")

    print(model_history)
    if args.result_name == None:
        pd.DataFrame(model_history).to_csv(f"results/{expriment_name}.csv")
    else:
        pd.DataFrame(model_history).to_csv(args.result_name)
