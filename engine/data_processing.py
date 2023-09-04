import os
import tarfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # * ImageNet distribution params
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

augmented_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.center_crop(224),
        transforms.ToTensor(),
        # * ImageNet distribution params
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def prepare_df(split_path: str, labels_Path: str, data_root: str) -> pd.DataFrame:
    def prepare_splits(indices: np.ndarray, value: int) -> pd.DataFrame:
        df = pd.Series(indices).value_counts().sort_index()
        df = (
            pd.DataFrame(df)
            .reset_index()
            .rename(columns={"index": "img_id", 0: "set_type"})
        )
        df["set_type"] = value
        return df

    def label_mat_to_df(path):
        data_set = loadmat(path)
        data = pd.DataFrame(data_set["labels"].reshape(-1), columns=["labels"])
        data.index = data.index + 1
        data.reset_index(inplace=True)
        data.rename(columns={"index": "img_id"}, inplace=True)

        return data

    def add_image_path(root: str, x: int) -> str:
        # * fill with zeros to the left max lenght = 5
        return os.path.join(root, f"image_{x:>05}.jpg")

    split = loadmat(split_path)
    # ** (1020,) (1020,) (6149,)
    # ** valid, tstid, trnid
    # ** dataset is flipped so we adjusted the correct title to the correct set
    train_df = prepare_splits(split["tstid"][0], 0)
    val_df = prepare_splits(split["valid"][0], 1)
    test_df = prepare_splits(split["trnid"][0], 2)
    split_df = pd.concat([train_df, val_df, test_df])

    labels = label_mat_to_df(labels_Path)

    merged = pd.merge(split_df, labels, on="img_id", how="inner")
    merged.sort_values(by=["img_id"], inplace=True)
    merged["image_path"] = merged["img_id"].apply(
        lambda x: add_image_path(data_root, x)
    )
    merged["image_path"] = merged["image_path"].apply(lambda x: x.replace("\\", "/"))

    train_split = merged[merged["set_type"] == 0]
    test_split = merged[merged["set_type"] == 2]
    val_split = merged[merged["set_type"] == 1]
    return train_split, test_split, val_split


class FlowerDataset(Dataset):
    def __init__(self, data_split: pd.DataFrame, transform: transforms = None):
        super().__init__()
        self.data_split = data_split
        self.transform = transform

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, index):
        img = Image.open(self.data_split.iloc[index]["image_path"])
        if self.transform:
            img = self.transform(img)
        classification = self.data_split.iloc[index]["labels"]
        classification = torch.tensor(classification, dtype=torch.long)
        return img, classification


def prepare_splits(
    split_path,
    labels_Path,
    data_root,
    transformations,
    train_batch_size,
    validation_batch_size,
    test_batch_size,
    shuffle_train,
    shuffle_validation,
    shuffle_test,
    **kwargs,
):
    train_split, test_split, val_split = prepare_df(split_path, labels_Path, data_root)
    train_dataset = FlowerDataset(train_split, transform=transformations)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=shuffle_train
    )

    val_dataset = FlowerDataset(val_split, transform=transformations)
    val_loader = DataLoader(
        val_dataset, batch_size=validation_batch_size, shuffle=shuffle_validation
    )

    test_dataset = FlowerDataset(test_split, transform=transformations)
    test_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=shuffle_test
    )

    return train_loader, val_loader, test_loader
