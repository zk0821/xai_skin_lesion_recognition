import abc
import torch
import torch.nn as nn
import torch.optim as optim
from average_meter import AverageMeter


class TemplateModel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, num_classes):
        super(TemplateModel, self).__init__()
        self.pretrained_model = self.load_pretrained()
        self.num_features = self.get_num_features_of_pretrained_model()
        self.final_layer = self.remove_final_layer()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear_one = nn.Linear(self.num_features, 512)
        self.linear_two = nn.Linear(512, 256)
        self.output = nn.Linear(256, num_classes)

    @abc.abstractmethod
    def remove_final_layer(self):
        return

    @abc.abstractmethod
    def get_num_features_of_pretrained_model(self):
        return

    @abc.abstractmethod
    def load_pretrained(self):
        return

    def forward(self, x):
        y = self.pretrained_model(x)
        y = self.flatten(y)
        y = self.linear_one(y)
        y = self.relu(y)
        y = self.linear_two(y)
        y = self.relu(y)
        y = self.output(y)
        return y

    def unfreeze_pretrained_layers(self):
        for param in self.pretrained_model.parameters():
            param.requires_grad = True
