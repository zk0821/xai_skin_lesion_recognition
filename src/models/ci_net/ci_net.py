import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms import v2
from torch.utils.data import DataLoader, WeightedRandomSampler
from early_stoppage import EarlyStoppage
from average_meter import AverageMeter
from cinet_model import Model
from ham10000_dataset import HAM10000Dataframe, HAM10000Dataset
from tqdm import tqdm
from torch.autograd import Variable
import ssl
import json
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CINet:
    def __init__(self, size=(224, 224), do_oversampling=False, do_loss_weights=False) -> None:
        print(f"Selected Model Type: CI-Net")
        self.model_type = "cinet"
        if do_oversampling:
            self.model_type += "_sampling"
        if do_loss_weights:
            self.model_type += "_weights"
        # Device
        self.device = torch.device("cuda")
        self.size = size
        self.do_oversampling = do_oversampling
        self.do_loss_weights = do_loss_weights

    def read_metadata_dataframe(self, path="data/ham10000/metadata.csv"):
        # Dataframe
        self.ham_df_object = HAM10000Dataframe(path="data/ham10000/metadata.csv")
        self.metadata_df = self.ham_df_object.get_dataframe()
        self.num_classes = self.metadata_df["type"].nunique()

    def perform_dataset_split(self):
        # Train - Validation - Test Split
        self.train_df, self.validation_and_test_df = train_test_split(
            self.metadata_df, test_size=0.3, random_state=42, stratify=self.metadata_df["type"]
        )
        self.validation_df, self.test_df = train_test_split(
            self.validation_and_test_df, test_size=0.5, random_state=42, stratify=self.validation_and_test_df["type"]
        )

    def create_dataloaders(self):
        # Dataloaders
        if self.do_oversampling:
            class_sample_count = np.array([len(np.where(self.train_df["type"] == t)[0]) for t in np.unique(self.train_df["type"])])
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in self.train_df["type"]])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            self.weighted_random_sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
        mean = [0.76303804, 0.54694057, 0.57165635]
        spread = [0.14156434, 0.15333284, 0.17053322]
        train_transforms = v2.Compose(
            [
                v2.Resize(self.size),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(20),
                v2.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(mean, spread),
            ]
        )
        train_dataset = HAM10000Dataset(self.train_df, img_dir="data/ham10000/images", transform=train_transforms)
        if self.do_oversampling:
            self.train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8, sampler=self.weighted_random_sampler)
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

        validation_transforms = v2.Compose(
            [
                v2.Resize(self.size),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(mean, spread),
            ]
        )
        validation_dataset = HAM10000Dataset(self.validation_df, img_dir="data/ham10000/images", transform=validation_transforms)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=8)

        test_transforms = v2.Compose(
            [
                v2.Resize(self.size),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(mean, spread),
            ]
        )
        test_dataset = HAM10000Dataset(self.test_df, img_dir="data/ham10000/images", transform=test_transforms)
        self.test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    def prepare_model(self):
        self.model = Model(self.num_classes)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0)
        if self.do_loss_weights:
            class_weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(self.train_df["type"]), y=self.train_df["type"].to_numpy()
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean").to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion0 = nn.CrossEntropyLoss().to(self.device)
        self.criterion1 = nn.L1Loss().to(self.device)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4)
        self.early_stoppage = EarlyStoppage(patience=10, min_delta=0.1)
        self.epoch_num = 200
        self.total_loss_train, self.total_acc_train = [], []
        self.total_loss_val, self.total_acc_val = [], []

    def load_model(self):
        self.model.load_state_dict(torch.load(f"models/{self.model_type}/{self.model_type}.pth"))

    def train_model(self):
        for epoch in range(1, self.epoch_num + 1):
            with tqdm(self.train_dataloader, unit="batch") as prog:
                prog.set_description(f"Epoch {epoch}/{self.epoch_num}")
                # Training
                self.model.train()
                train_loss = AverageMeter()
                train_acc = AverageMeter()
                for _, data in enumerate(self.train_dataloader):
                    prog.update()
                    (img0, img1, label0, label1, syn_label, dislabel11, dislabel12, dislabel21, dislabel22) = data
                    N = img0.size(0)
                    # Images
                    img0 = Variable(img0).to(self.device)
                    img1 = Variable(img1).to(self.device)
                    # Labels
                    class_label1 = label0.to(self.device)
                    class_label2 = label1.to(self.device)
                    # SynLabels
                    synlabel = syn_label.to(self.device)
                    # DisLabels
                    dislabel11 = dislabel11.to(self.device)
                    dislabel12 = dislabel12.to(self.device)
                    dislabel21 = dislabel21.to(self.device)
                    dislabel22 = dislabel22.to(self.device)

                    result1, result2, synscore = self.model(img0, img1)
                    loss_syn = self.criterion0(synscore, synlabel)
                    loss_ce1 = self.criterion(result1, class_label1)
                    loss_ce2 = self.criterion(result2, class_label2)

                    loss = loss_ce1 + loss_ce2 + 0.1 * loss_syn

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    _, predictions = result1.max(1, keepdim=True)
                    curr_acc = predictions.eq(class_label1.view_as(predictions)).sum().item() / N
                    train_acc.update(curr_acc)
                    curr_loss = loss.item()
                    train_loss.update(curr_loss)
                    prog.set_postfix({"Train Accuracy": curr_acc, "Train Loss": curr_loss, "Validation Accuracy": np.NAN, "Validation Loss": np.NAN})
                self.total_loss_train.append(train_loss.avg)
                self.total_acc_train.append(train_acc.avg)
                # Validation
                loss_val, acc_val = self.evaluate_model(self.validation_dataloader)
                self.total_loss_val.append(loss_val)
                self.total_acc_val.append(acc_val)
                prog.set_postfix(
                    {"Train Accuracy": train_acc.avg, "Train Loss": train_loss.avg, "Validation Accuracy": acc_val, "Validation Loss": loss_val}
                )
                prog.refresh()
                # Scheduler
                self.scheduler.step(loss_val)
                # Model Checkpoint
                if loss_val < self.early_stoppage.get_min_validation_loss():
                    print(
                        f"Best validation loss detected at epoch {epoch}! Validation Loss: {loss_val}, Validation Accuracy: {acc_val}. Saving model.."
                    )
                    torch.save(self.model.state_dict(), f"models/{self.model_type}/{self.model_type}.pth")
                # Early Stoppage
                if self.early_stoppage.early_stop(loss_val):
                    print(f"Early stoppage...")
                    break

    def evaluate_model(self, dataloader):
        self.model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        with torch.no_grad():
            for ix, data in enumerate(dataloader):
                (img0, img1, label0, label1, syn_label, dislabel11, dislabel12, dislabel21, dislabel22) = data
                N = img0.size(0)
                # Images
                img0 = Variable(img0).to(self.device)
                img1 = Variable(img1).to(self.device)
                # Labels
                class_label1 = label0.to(self.device)
                class_label2 = label1.to(self.device)
                # SynLabels
                synlabel = syn_label.to(self.device)
                # DisLabels
                dislabel11 = dislabel11.to(self.device)
                dislabel12 = dislabel12.to(self.device)
                dislabel21 = dislabel21.to(self.device)
                dislabel22 = dislabel22.to(self.device)

                result1, result2, synscore = self.model(img0, img1)
                _, predictions = result1.max(1, keepdim=True)

                val_acc.update(predictions.eq(class_label1.view_as(predictions)).sum().item() / N)
                val_loss.update(self.criterion(result1, class_label1).item())
                # ROC AUC
                if ix == 0:
                    all_labels = class_label1.cpu().numpy()
                    all_predictions = predictions.cpu().numpy()
                    all_probs = torch.softmax(result1, dim=1).cpu().numpy()
                else:
                    all_labels = np.concatenate((all_labels, class_label1.cpu().numpy()), axis=0)
                    all_predictions = np.concatenate((all_predictions, predictions.cpu().numpy()), axis=0)
                    all_probs = np.concatenate((all_probs, torch.softmax(result1, dim=1).cpu().numpy()), axis=0)
            auc_macro_ovr = roc_auc_score(all_labels, all_probs, average="macro", multi_class="ovr")
            auc_macro_ovo = roc_auc_score(all_labels, all_probs, average="macro", multi_class="ovo")
            ap_macro = average_precision_score(all_labels, all_probs, average="macro")
            print(f"AUC Macro OVR: {auc_macro_ovr}, AUC Macro OVO: {auc_macro_ovo}, AP Macro: {ap_macro}")
            # Build confusion matrix
            cf_matrix = confusion_matrix(all_labels, all_predictions)
            classes = self.ham_df_object.get_categories().categories
            new_classes = []
            for clss in classes:
                new_class = clss.replace(" ", "\n")
                new_classes.append(new_class)
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in new_classes], columns=[i for i in new_classes])
            plt.figure(figsize=(15, 10))
            sns.heatmap(df_cm, annot=True)
            plt.savefig(f"models/{self.model_type}/confusion_matrix.png")
        return val_loss.avg, val_acc.avg

    def test_model(self):
        loss_test, acc_test = self.evaluate_model(self.test_dataloader)
        print(f"Test Loss: {loss_test}, Test Accuracy: {acc_test}")

    def dump_metrics(self):
        # Save training data
        with open(f"models/{self.model_type}/train_acc.txt", "w") as train_acc_file:
            json.dump(self.total_acc_train, train_acc_file)
        with open(f"models/{self.model_type}/train_loss.txt", "w") as train_loss_file:
            json.dump(self.total_loss_train, train_loss_file)
        with open(f"models/{self.model_type}/val_acc.txt", "w") as val_acc_file:
            json.dump(self.total_acc_val, val_acc_file)
        with open(f"models/{self.model_type}/val_loss.txt", "w") as val_loss_file:
            json.dump(self.total_loss_val, val_loss_file)


def main(train=False, sampling=False, class_weights=False):
    # For downloading models from PyTorch Vision
    ssl._create_default_https_context = ssl._create_unverified_context
    cinet = CINet(do_oversampling=sampling, do_loss_weights=class_weights)
    cinet.read_metadata_dataframe(path="data/ham10000/metadata.csv")
    cinet.perform_dataset_split()
    cinet.create_dataloaders()
    cinet.prepare_model()
    if train:
        cinet.train_model()
        cinet.load_model()
        cinet.test_model()
        cinet.dump_metrics()
    else:
        cinet.load_model()
        cinet.test_model()
    print("Finished!")


if __name__ == "__main__":
    main(train=True, sampling=False, class_weights=True)
