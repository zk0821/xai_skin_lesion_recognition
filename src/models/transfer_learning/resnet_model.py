from template_model import TemplateModel
import torch
import torch.nn as nn


class CustomResNet(TemplateModel):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def load_pretrained(self):
        return torch.hub.load("pytorch/vision", "resnet152", weights="IMAGENET1K_V2")

    def remove_final_layer(self):
        self.pretrained_model.fc = nn.Sequential()

    def get_num_features_of_pretrained_model(self):
        return self.pretrained_model.fc.in_features

    def forward(self, x):
        return super().forward(x)

    def unfreeze_pretrained_layers(self):
        super().unfreeze_pretrained_layers()
