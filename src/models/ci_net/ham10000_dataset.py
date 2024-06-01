from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import pandas as pd
import os
import torch
from random import randint


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None) -> None:
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.length = len(self.dataframe)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Image One
        img_one_path = f"{self.img_dir}/{self.dataframe['isic_id'].iloc[idx]}.jpg"
        image_one = Image.open(img_one_path)
        label_one = torch.tensor(int(self.dataframe["type"].iloc[idx]))
        # Image two
        if idx != self.length - 1:
            a = randint(1, self.length - idx - 1)
            new_index = idx + a
        else:
            a = randint(1, self.length - 1)
            new_index = idx - a
        img_two_path = f"{self.img_dir}/{self.dataframe['isic_id'].iloc[new_index]}.jpg"
        image_two = Image.open(img_two_path)
        label_two = torch.tensor(int(self.dataframe["type"].iloc[new_index]))
        # Syn Label
        if label_one == label_two:
            syn_label = 1
        else:
            syn_label = 0
        # Dis labels
        dislabel11 = 1
        dislabel12 = 0
        dislabel21 = 0
        dislabel22 = 1
        # Transformations
        if self.transform is not None:
            image_one = self.transform(image_one)
            image_two = self.transform(image_two)
        # Returns
        return image_one, image_two, label_one, label_two, syn_label, dislabel11, dislabel12, dislabel21, dislabel22


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
