from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import pandas as pd
import os
import torch


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None) -> None:
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.dataframe['isic_id'].iloc[idx]}.jpg"
        image = Image.open(img_path)
        label = torch.tensor(int(self.dataframe["type"].iloc[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class HAM10000VerboseDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None) -> None:
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.dataframe['isic_id'].iloc[idx]}.jpg"
        image = Image.open(img_path)
        label = torch.tensor(int(self.dataframe["type"].iloc[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return img_path, image, label


class HAM10000Dataframe:
    def __init__(self, path="data/ham10000/metadata.csv"):
        # Read the csv file
        self.metadata_df = pd.read_csv(path)
        # Add categorical type
        self.categories = pd.Categorical(self.metadata_df["diagnosis"])
        self.metadata_df["type"] = self.categories.codes
        # Assertions
        assert self.metadata_df["isic_id"].duplicated().any() == False  # No duplicates

    def get_dataframe(self):
        return self.metadata_df

    def get_categories(self):
        return self.categories

    def print_diagnosis_counts(self):
        grouped_df = self.metadata_df.groupby("diagnosis")
        print(grouped_df["diagnosis"].count())
