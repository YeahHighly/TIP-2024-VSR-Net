from dataloader.drive import get_drive_dataloader  
from dataloader.octa import get_octa_dataloader  

class DataFactory:
    def __init__(self):
        self.available_datasets = {
            "drive": self.get_drive,
            "octa": self.get_drive
        }

    def get_drive(self, batch_size=4, train=True):
        root_dir = "../dataset/DRIVE_AV/"  # Replace with your dataset path
        train_loader = get_drive_dataloader(root_dir, batch_size=4, train=True)
        return train_loader
    
    def get_octa(self, batch_size=4, train=True):
        root_dir = "../dataset/OCTA-500/"  # Replace with your dataset path
        train_loader = get_drive_dataloader(root_dir, batch_size=4, train=True)
        return train_loader

    def get_dataset(self, dataset_name, batch_size=4, train=True):

        dataset_name = dataset_name.lower()
        if dataset_name in self.available_datasets:
            return self.available_datasets[dataset_name](batch_size, train)
        else:
            raise ValueError(f"Unknown dataset: {model_name}. Available datasets: {list(self.available_models.keys())}")

