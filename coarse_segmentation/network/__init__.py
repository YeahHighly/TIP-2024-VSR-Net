import torch
import torch.nn as nn
from torchvision import models
from network.unet import UNet  
from network.cenet import CE_Net
from network.csnet import CSNet
from network.skelcon import LUNet

class ModelFactory:
    def __init__(self):
        self.available_models = {
            "unet": self.get_unet,
            "cenet": self.get_cenet,
            "csnet": self.get_csnet,
            "skelcon": self.get_skelcon
        }

    def get_unet(self, num_classes=1, num_channels=3):

        return UNet(n_channels=num_channels, n_classes=num_classes)

    def get_cenet(self, num_classes=1, num_channels=3):

        return CE_Net(n_channels=num_channels, n_classes=num_classes)

    def get_csnet(self, num_classes=1, num_channels=3):

        return CSNet(n_channels=num_channels, n_classes=num_classes)

    def get_skelcon(self, num_classes=1, num_channels=3):

        return LUNet(n_channels=num_channels, n_classes=num_classes)

    def get_model(self, model_name, num_classes=1, num_channels=3):

        model_name = model_name.lower()
        if model_name in self.available_models:
            return self.available_models[model_name](num_classes, num_channels)
        else:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.available_models.keys())}")

