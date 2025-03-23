# encoding: utf-8
import os
import torch
import argparse
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import network

# ------------------------ Argument Parser ------------------------
parser = argparse.ArgumentParser(description="coarse_segmentation_prediction")
parser.add_argument("--network", type=str, default='cenet', help="cenet, unet, csnet, skelcon")
parser.add_argument("--dataset", type=str, default='drive', help="drive, octa") 
parser.add_argument("--set", type=str, default='training', help="training or testing")
parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device index.")
parser.add_argument("--model_path", type=str, default="./checkpoints/", help="Path to saved model checkpoint.")
args = parser.parse_args()

# ------------------------ Root Paths ------------------------
root_drive_path = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/DRIVE_AV/"
root_octa_path = "/home/hly/iMED-Code/TIP-2024-VSR-Net/dataset/OCTA-500/"

# ------------------------ Path Setup Based on Dataset and Set ------------------------
if args.dataset.lower() == 'drive':
    image_path = os.path.join(root_drive_path, args.set, 'images')
    save_path = os.path.join(root_drive_path, args.set, 'coarse')
elif args.dataset.lower() == 'octa':
    image_path = os.path.join(root_octa_path, 'OCTAFULL')
    save_path = os.path.join(root_octa_path, 'coarse')
else:
    raise ValueError("Unsupported dataset. Use 'drive' or 'octa'.")

# ------------------------ Environment Configuration ------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directory exists
os.makedirs(save_path, exist_ok=True)

# Image size configuration
IMAGE_SIZE = (512, 512)

def preprocess_image(image_path):
    """Load and preprocess the image using albumentations and OpenCV."""
    image = cv2.imread(image_path)
    original_size = (image.shape[1], image.shape[0])  # Store original image size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(*IMAGE_SIZE),
        ToTensorV2()
    ])

    image = transform(image=image)["image"].unsqueeze(0)
    return image, original_size

# ------------------------ Prediction Function ------------------------
def predict():
    print("Initializing Prediction...")

    # Initialize the model
    model_factory = network.ModelFactory()
    model = model_factory.get_model(args.network, num_classes=1, num_channels=3)
    model = model.to(DEVICE)

    # Load pre-trained weights
    weight_file = os.path.join(args.model_path, args.dataset, args.network, "best.pth")
    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location=DEVICE, weights_only=True))
        print(f"Model weights loaded from {weight_file}")
    else:
        raise FileNotFoundError(f"Weight file not found: {weight_file}")
    
    model.eval()
    
    print("\nPredicting and saving results...")
    image_files = [f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.tif', '.bmp'))]
    
    with torch.no_grad():
        for filename in image_files:
            img_fp = os.path.join(image_path, filename)
            image, original_size = preprocess_image(img_fp)
            image = image.to(DEVICE)
            
            output = model(image.float())
            pred = output.cpu().numpy()[0, 0] * 255  # Convert to grayscale image
            pred_img = pred.astype(np.uint8)
            
            # Resize prediction back to original size using nearest-neighbor interpolation
            pred_resized = cv2.resize(pred_img, original_size, interpolation=cv2.INTER_NEAREST)
            
            save_filepath = os.path.join(save_path, filename[:-4]+'.png')
            cv2.imwrite(save_filepath, pred_resized)
            print(f"Saved: {save_filepath}")
    
if __name__ == "__main__":
    predict()
