import torch
import torch.nn as nn
from resnet_model import CustomResNet
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchvision import models
from skimage.io import imread
from skimage.transform import resize
from torchvision.transforms import v2
from ham10000_dataset import HAM10000Dataframe, HAM10000VerboseDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import cv2


class GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None

        # PRETRAINED MODEL
        self.pretrained = CustomResNet(8)
        self.pretrained.load_state_dict(torch.load(f"models/resnet/resnet.pth"))

        self.features_conv = nn.Sequential(
            self.pretrained.pretrained_model.conv1,
            self.pretrained.pretrained_model.bn1,
            self.pretrained.pretrained_model.relu,
            self.pretrained.pretrained_model.maxpool,
            self.pretrained.pretrained_model.layer1,
            self.pretrained.pretrained_model.layer2,
            self.pretrained.pretrained_model.layer3,
            self.pretrained.pretrained_model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            self.pretrained.flatten,
            self.pretrained.linear_one,
            self.pretrained.relu,
            self.pretrained.linear_two,
            self.pretrained.relu,
            self.pretrained.output,
        )

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


def main():
    # Dataframe
    ham_df_object = HAM10000Dataframe(path="data/ham10000/metadata.csv")
    metadata_df = ham_df_object.get_dataframe()
    num_classes = metadata_df["type"].nunique()
    # Train - Validation - Test Split
    train_df, validation_and_test_df = train_test_split(metadata_df, test_size=0.3, random_state=42, stratify=metadata_df["type"])
    validation_df, test_df = train_test_split(validation_and_test_df, test_size=0.5, random_state=42, stratify=validation_and_test_df["type"])
    mean = [0.76303804, 0.54694057, 0.57165635]
    spread = [0.14156434, 0.15333284, 0.17053322]
    test_transforms = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean, spread),
        ]
    )
    test_dataset = HAM10000VerboseDataset(test_df, img_dir="data/ham10000/images", transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    # Model
    model = GradCamModel()
    model.eval()
    # get the image from the dataloader
    for ix, data in enumerate(test_dataloader):
        if ix == 10:
            (img_path, img, label) = data
            break
    print(f"Image Path: {img_path[0]}")

    # get the most likely prediction of the model
    pred = model(img)

    # get the gradient of the output with respect to the parameters of the model
    pred[:, label].backward()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.get_activations(img).detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    plt.matshow(heatmap.squeeze())
    plt.savefig("heatmap.jpg")
    img = cv2.imread(img_path[0])
    cv2.imwrite("original.jpg", img)
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite("gradcam.jpg", superimposed_img)


if __name__ == "__main__":
    main()
