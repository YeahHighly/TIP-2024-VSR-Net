import torch
import torch.nn as nn
from torchvision import models
from module.ccm import CCMBase  
from module.ccm_plus import CMMPlus  
from module.cmm import CMMBase  

class CCMFactory:
    def __init__(self):
        self.available_models = {
            "ccm": self.get_ccm,
            "ccm_plus": self.get_ccm_plus,
        }

    def get_ccm(self, num_classes):

        return CCMBase(in_channels=512, hidden_channels=256, num_classes=num_classes)

    def get_ccm_plus(self, num_classes):

        return CMMPlus(in_channels=512, hidden_channels=256, num_heads=8, num_classes=num_classes)

    def get_model(self, model_name, num_classes=2):

        model_name = model_name.lower()
        if model_name in self.available_models:
            return self.available_models[model_name](num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.available_models.keys())}")


class CMMFactory:
    def __init__(self):
        self.available_models = {
            "cmm": self.get_cmm
        }

    def get_cmm(self):

        return CMMBase(base_features=64)

    def get_model(self, model_name):

        model_name = model_name.lower()
        if model_name in self.available_models:
            return self.available_models[model_name]()
        else:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.available_models.keys())}")

