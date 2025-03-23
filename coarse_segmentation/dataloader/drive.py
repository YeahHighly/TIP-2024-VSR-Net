import os 
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

class DRIVEDataset(Dataset):
    """
    DRIVE dataset for vessel segmentation.
    """
    def __init__(self, root_dir, train=True):
        """
        Args:
            root_dir (str): Path to the DRIVE dataset.
            train (bool): Whether to load training or test set.
        """
        self.root_dir = root_dir
        self.train = train
        self.size = 512

        # Define paths for images and masks
        self.image_dir = os.path.join(root_dir, "training/images" if train else "testing/images")
        self.mask_dir = os.path.join(root_dir, "training/vessel" if train else "testing/vessel")

        # List all images and masks
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".tif")])
        self.mask_filenames = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(".png")])

        # Define transformations
        if train:
            self.transform = A.Compose([
                A.Resize(self.size, self.size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.size, self.size),
                ToTensorV2()
            ])

        assert len(self.image_filenames) == len(self.mask_filenames), "Number of images and masks must match!"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        mask = mask.float() / 255.0
        image = image.float()
        return image, mask.unsqueeze(0)

# Create dataset & dataloader
def get_drive_dataloader(root_dir, batch_size=4, train=True, shuffle=True, num_workers=2):
    dataset = DRIVEDataset(root_dir=root_dir, train=train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Example usage:
if __name__ == "__main__":
    root_dir = "../../dataset/DRIVE_AV/"  # Replace with your dataset path
    train_loader = get_drive_dataloader(root_dir, batch_size=4, train=True)

    # Test the dataloader
    for images, masks in train_loader:
        print("Image batch shape:", images.shape)  # (B, 3, 512, 512)
        print("Mask batch shape:", masks.shape)    # (B, 1, 512, 512)
        break
