import os 
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import random
import glob
random.seed(1024)

class OCTADataset(Dataset):

    def __init__(self, root_dir, train=True):

        self.root_dir = root_dir
        self.train = train
        self.size = 512

        # Define paths for images and masks
        self.image_dir = os.path.join(root_dir, "OCTAFULL")

        # List all images and masks
        self.image_filenames = glob.glob(self.image_dir+"/*.bmp")
        random.shuffle(self.image_filenames)

        # Define transformations
        if train:
            self.transform = A.Compose([
                A.Resize(self.size, self.size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                ToTensorV2()
            ])
            self.image_filenames = self.image_filenames[:int(len(self.image_filenames)*0.8)]
        else:
            self.transform = A.Compose([
                A.Resize(self.size, self.size),
                ToTensorV2()
            ])
            self.image_filenames = self.image_filenames[int(len(self.image_filenames)*0.8):]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.image_filenames[idx]
        mask_path = self.image_filenames[idx].replace("OCTAFULL", "GroundTruth")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask==255, 255, 0).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        mask = mask.float() / 255.0
        image = image.float()
        return image, mask.unsqueeze(0)

# Create dataset & dataloader
def get_octa_dataloader(root_dir, batch_size=4, train=True, shuffle=True, num_workers=2):
    dataset = OCTADataset(root_dir=root_dir, train=train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Example usage:
if __name__ == "__main__":
    root_dir = "../../dataset/OCTA-500/"  # Replace with your dataset path
    train_loader = get_octa_dataloader(root_dir, batch_size=4, train=True)

    # Test the dataloader
    for images, masks in train_loader:
        print("Image batch shape:", images.shape)  # (B, 3, 512, 512)
        print("Mask batch shape:", masks.shape)    # (B, 1, 512, 512)
        break
