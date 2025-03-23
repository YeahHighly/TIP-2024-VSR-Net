# encoding: utf-8
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import network

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/hly/iMED-Code/TIP-2024-VSR-Net/coarse_segmentation/checkpoints/drive/cenet/best.pth"  
IMAGE_PATH = "/home/hly/Data/DRIVE_AV/testing/images/01_test.tif" 
SAVE_PATH = "inference.png"
IMAGE_SIZE = (512, 512)  

def preprocess_image(image_path):
    """Load and preprocess the image using albumentations and OpenCV."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(*IMAGE_SIZE),
        # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    image = transform(image=image)["image"].unsqueeze(0)
    return image

def load_model():
    """Load the trained vessel segmentation model."""
    model_factory = network.ModelFactory()
    model = model_factory.get_model("cenet", num_classes=1, num_channels=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def inference():
    """Perform inference and visualize the results."""
    print("Loading model...")
    model = load_model()

    print(f"Reading image: {IMAGE_PATH}")
    image = preprocess_image(IMAGE_PATH).to(DEVICE)

    with torch.no_grad():
        output = model(image.float()).squeeze().cpu().numpy()
        prediction = (output > 0.5).astype(np.uint8)

    mask_path = IMAGE_PATH.replace("images", "vessel").replace(".tif", ".png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMAGE_SIZE)

    print("Inference completed. Saving result...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    original = cv2.imread(IMAGE_PATH)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, IMAGE_SIZE)
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Predicted mask
    axes[1].imshow(prediction, cmap="gray")
    axes[1].set_title("Predicted Vessel Mask")
    axes[1].axis("off")

    # Ground truth mask
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("GT Vessel Mask")
    axes[2].axis("off")

    plt.savefig(SAVE_PATH)
    print(f"Result saved to {SAVE_PATH}")

if __name__ == "__main__":
    inference()