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
from engine.train import training
from engine.utils import accuray_fn, load_configs, set_global_seed
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")


configs = load_configs()

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
        freeze=FREEZE_RESNET,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        epoch_train_loss, epoch_train_acc = 0, 0
        epoch_val_loss, epoch_val_acc = 0, 0
        for image_batch, label_batch in tqdm(
            train_loader, total=len(train_loader), desc=f"Epoch {epoch}", leave=True
        ):
            train_loss, train_acc = training(
                model,
                loss_fn,
                optimizer,
                image_batch,
                label_batch,
                DEVICE,
                step_type="train",
            )
            epoch_train_loss += train_loss
            epoch_train_acc += train_acc

        for image_batch, label_batch in tqdm(
            val_loader, total=len(val_loader), desc=f"Epoch {epoch}", leave=True
        ):
            val_loss, val_acc = training(
                model,
                loss_fn,
                optimizer,
                image_batch,
                label_batch,
                DEVICE,
                step_type="val",
            )
            epoch_val_loss += val_loss
            epoch_val_acc += val_acc

        # * Printing the results

    #         x, y = x.to(device), (y - 1).to(device)

    #         optimizer.zero_grad()
    #         y_hat = model(x)
    #         loss = loss_fn(y_hat, y)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         total_acc += accuray_fn(y_hat, y)

    #     # writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
    #     # writer.add_scalar("Accuracy/train", total_acc / len(train_loader), epoch)

    #     print(f"Epoch {epoch} Loss : {total_loss / len(train_loader)}")
    #     print(f"Epoch {epoch} Accuracy: {total_acc/len(train_loader)}")
