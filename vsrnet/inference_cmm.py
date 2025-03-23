import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import argparse
from torchvision.utils import save_image
import module
import dataloader
import cv2

# ------------------------ Argument Parser ------------------------
parser = argparse.ArgumentParser(description="Inference for CMM Segmentation")
parser.add_argument("--module", type=str, default='cmm', help="CMM segmentation backbone.")
parser.add_argument("--dataset", type=str, default='drive', help="Dataset name.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
parser.add_argument("--model_path", type=str, default="./checkpoints/", help="Path to the trained model.")
parser.add_argument("--result_path", type=str, default="./cmm_result/", help="Path to save predictions.")
args = parser.parse_args()

# ------------------------ Environment Configuration ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.result_path, exist_ok=True)

# ------------------------ Load Model ------------------------
model_factory = module.CMMFactory()
model = model_factory.get_model(args.module)
model.load_state_dict(torch.load(os.path.join(args.model_path, args.dataset, args.module, "best.pth"), map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# ------------------------ Load Data ------------------------
factory = dataloader.CMMDataFactory()
test_loader = factory.get_dataset(args.dataset, batch_size=args.batch_size, train=False)

# ------------------------ Inference ------------------------
with torch.no_grad():
    images, mappings, clusters = next(iter(test_loader))  # Get one batch
    images, mappings = images.to(DEVICE), mappings.to(DEVICE)
    outputs = model(images, mappings)
    preds = (outputs > 0.5).float().cpu().numpy() * 255
    
    mappings = mappings.cpu().numpy() * 255
    clusters = clusters.cpu().numpy() * 255
    
    # Save results
    for i in range(preds.shape[0]):
        pred_img = preds[i].squeeze().astype(np.uint8)
        mapping_img = mappings[i].squeeze().astype(np.uint8)
        cluster_img = clusters[i].squeeze().astype(np.uint8)
        
        # Horizontally stack the images
        combined_img = np.hstack([pred_img, mapping_img, cluster_img])
        save_path = os.path.join(args.result_path, f"result_{i}.png")
        cv2.imwrite(save_path, combined_img)
        print(f"Saved: {save_path}")

print("Inference completed. Results saved in result folder.")
