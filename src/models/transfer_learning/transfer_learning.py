from resnet_model import CustomResNet
from densenet_model import CustomDenseNet
from efficientnet_model import CustomEfficientNet
from ham10000_dataset import HAM10000Dataset, HAM10000Dataframe
from average_meter import AverageMeter
from early_stoppage import EarlyStoppage
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms import v2
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import ssl
import cv2
from tqdm import tqdm
import glob
import json


class TransferLearning:
    def __init__(self, model_type, size=(224, 224), do_oversampling=False, do_loss_weights=False) -> None:
        # Device
        self.device = torch.device("cuda")
        # Params
        if model_type == "resnet":
            print(f"Selected Model Type: ResNet")
            self.model_type = model_type
        elif model_type == "densenet":
            print(f"Selected Model Type: DenseNet")
            self.model_type = model_type
        elif model_type == "efficientnet":
            print(f"Selected Model Type: EfficientNet")
            self.model_type = model_type
        else:
            raise RuntimeError(f"Model type: {model_type} is not supported!")
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
        if self.model_type == "resnet":
            self.model = CustomResNet(num_classes=self.num_classes)
        elif self.model_type == "densenet":
            self.model = CustomDenseNet(num_classes=self.num_classes)
        elif self.model_type == "efficientnet":
            self.model = CustomEfficientNet(num_classes=self.num_classes)
        else:
            raise RuntimeError(f"Unsupported model type!")
        self.model.unfreeze_pretrained_layers()
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
                    images, labels = data
                    N = images.size(0)

                    images = Variable(images).to(self.device)
                    labels = Variable(labels).to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(images)

                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    _, predictions = outputs.max(1, keepdim=True)
                    curr_acc = predictions.eq(labels.view_as(predictions)).sum().item() / N
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
                images, labels = data
                N = images.size(0)

                images = Variable(images).to(self.device)
                labels = Variable(labels).to(self.device)

                outputs = self.model(images)
                _, predictions = outputs.max(1, keepdim=True)

                val_acc.update(predictions.eq(labels.view_as(predictions)).sum().item() / N)
                val_loss.update(self.criterion(outputs, labels).item())
                # ROC AUC
                if ix == 0:
                    all_labels = labels.cpu().numpy()
                    all_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                else:
                    all_labels = np.concatenate((all_labels, labels.cpu().numpy()), axis=0)
                    all_probs = np.concatenate((all_probs, torch.softmax(outputs, dim=1).cpu().numpy()), axis=0)
            auc_micro = roc_auc_score(all_labels, all_probs, average="micro", multi_class="ovr")
            ap_micro = average_precision_score(all_labels, all_probs, average="micro")
            print(f"AUC Micro: {auc_micro}, AP Micro: {ap_micro}")
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
