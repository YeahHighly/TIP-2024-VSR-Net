from dataloader.drive_refine_ccm import get_drive_ccm_dataloader  
from dataloader.octa_refine_ccm import get_octa_ccm_dataloader  
from dataloader.drive_refine_cmm import get_drive_cmm_dataloader  
from dataloader.octa_refine_cmm import get_octa_cmm_dataloader  


class CCMDataFactory:
    def __init__(self):
        self.available_datasets = {
            "drive": self.get_drive_ccm,
            "cota": self.get_octa_ccm
        }

    def get_drive_ccm(self, batch_size=4, train=True):
        if train:
            graph_dir = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/DRIVE_AV/testing/graph_data/" 
        else:
            graph_dir = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/DRIVE_AV/training/graph_data/" 
        train_loader = get_drive_ccm_dataloader(graph_dir, batch_size=batch_size)
        return train_loader
    
    def get_octa_ccm(self, batch_size=4, train=True):
        graph_dir = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/OCTA-500/graph_data/" 
        train_loader = get_octa_ccm_dataloader(graph_dir, train, batch_size=batch_size)
        return train_loader

    def get_dataset(self, dataset_name, batch_size=4, train=True):

        dataset_name = dataset_name.lower()
        if dataset_name in self.available_datasets:
            return self.available_datasets[dataset_name](batch_size, train)
        else:
            raise ValueError(f"Unknown dataset: {model_name}. Available datasets: {list(self.available_models.keys())}")

class CMMDataFactory:
    def __init__(self):
        self.available_datasets = {
            "drive": self.get_drive_cmm,
            "octa": self.get_octa_cmm,
        }

    def get_drive_cmm(self, batch_size=12, train=True):

        root_dir = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/DRIVE_AV/" 
        train_loader = get_drive_cmm_dataloader(root_dir, batch_size=batch_size)
        return train_loader
    
    def get_octa_cmm(self, batch_size=12, train=True):

        root_dir = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/OCTA-500/" 
        train_loader = get_octa_cmm_dataloader(root_dir, batch_size=batch_size)
        return train_loader

    def get_dataset(self, dataset_name, batch_size=12, train=True):

        dataset_name = dataset_name.lower()
        if dataset_name in self.available_datasets:
            return self.available_datasets[dataset_name](batch_size, train)
        else:
            raise ValueError(f"Unknown dataset: {model_name}. Available datasets: {list(self.available_models.keys())}")

