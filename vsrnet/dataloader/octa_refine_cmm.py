# encoding: utf-8
import os
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import glob

class OCTA_CMM_Dataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        self.patch_size = 256

        self.rehabilitate_dir = os.path.join(root_dir, "rehabilitate/")
        self.image_filenames = glob.glob(self.rehabilitate_dir+"*roi*.png")

        if self.train:
            self.image_filenames = self.image_filenames[:int(len(self.image_filenames)*0.8)]
        else:
            self.image_filenames = self.image_filenames[int(len(self.image_filenames)*0.8):]

        assert len(self.image_filenames) > 0, "Mismatch in file counts!"

        if self.train:
            self.transform = A.Compose([
                A.Resize(self.patch_size, self.patch_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                ToTensorV2()
            ], additional_targets={"mask2": "mask"})
        else:
            self.transform = A.Compose([
                A.Resize(self.patch_size, self.patch_size),
                ToTensorV2()
            ], additional_targets={"mask2": "mask"})

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        cluster_path = img_path.replace('roi', 'cluster')
        map_path = img_path.replace('roi', 'map')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cluster = cv2.imread(cluster_path, cv2.IMREAD_GRAYSCALE)
        mapping = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        combined_pairs = np.stack([mapping, cluster], axis=-1)

        transformed = self.transform(image=image, mask=combined_pairs)
        image = transformed["image"]
        mapping = transformed["mask"][..., 0]
        cluster = transformed["mask"][..., 1]

        image_patches = image.float()
        mapping_patches = mapping.unsqueeze(0).float() / 255.0
        cluster_patches = cluster.unsqueeze(0).float() / 255.0

        return image_patches, mapping_patches, cluster_patches

# def _collate_fn(batch):
#     images, mapping, cluster = zip(*batch)

#     images = torch.stack(images)
#     mapping = torch.stack(mapping)
#     cluster = torch.stack(cluster)

#     batch_size, num_patches, c, h, w = images.shape
#     images = images.view(batch_size * num_patches, c, h, w)
#     mapping = mapping.view(batch_size * num_patches, 1, h, w)
#     cluster = cluster.view(batch_size * num_patches, 1, h, w)

#     return images, mapping, cluster

def get_octa_cmm_dataloader(root_dir, batch_size=4, train=True, shuffle=True, num_workers=2):
    dataset = OCTA_CMM_Dataset(root_dir=root_dir, train=train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers)
    return dataloader

def test_cmm_dataset():
    root_dir = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/OCTA-500/"
    # save_dir = "./reconstructed_results/"
    train_loader = get_octa_cmm_dataloader(root_dir, batch_size=4, train=True)
    
    for images, mapping, cluster in train_loader:
        print(images.shape, mapping.shape, cluster.shape)
        print(torch.max(images), torch.max(mapping), torch.max(cluster))
        break

if __name__ == "__main__":
    test_cmm_dataset()
