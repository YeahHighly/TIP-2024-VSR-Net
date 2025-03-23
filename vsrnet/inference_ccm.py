import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse
import numpy as np
from torch.nn import functional as F
import module
import dataloader

# ------------------------ Argument Parser ------------------------
parser = argparse.ArgumentParser(description="Graph Edge Classification - Inference")
parser.add_argument("--module", type=str, default='ccm_plus', help="Graph classification backbone, 'ccm' and 'ccm_plus'")
parser.add_argument("--dataset", type=str, default='drive', help="Graph dataset.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing.")
parser.add_argument("--save_path", type=str, default="./checkpoints", help="Path to saved model checkpoints.")
args = parser.parse_args()

# ------------------------ Environment Configuration ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.save_path = os.path.join(args.save_path, args.dataset, args.module)
model_path = os.path.join(args.save_path, "best.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

# ------------------------ Load Test Data ------------------------
datafactory = dataloader.CCMDataFactory()
test_loader = datafactory.get_dataset(args.dataset, batch_size=args.batch_size, train=False)

# Get input feature size and number of classes
sample_data = next(iter(test_loader))
in_channels = sample_data[0].x.shape[1]
num_classes = 2

# ------------------------ Load Model ------------------------
ccmfactory = module.CCMFactory()
model = ccmfactory.get_model(args.module, num_classes=1)
state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

# ------------------------ Predict Five Samples ------------------------
predictions = []
ground_truths = []

with torch.no_grad():
    count = 0
    for batch in test_loader:
        batch = [data.to(DEVICE) for data in batch]
        for data in batch:
            if count >= 5:
                break  # Stop after five samples
            outputs = model(data)
            preds = (outputs >= 0.5).int().cpu().numpy()  # Convert probabilities to binary predictions.cpu().numpy()
            gt = data.edge_label.cpu().numpy()
            
            predictions.append(preds)
            ground_truths.append(gt)
            count += 1
        if count >= 5:
            break

# ------------------------ Output Results ------------------------
for i in range(len(predictions)):
    print(f"Sample {i + 1}:")
    print(f"  Predicted: {predictions[i]}")
    print(f"  Ground Truth: {ground_truths[i]}")
    print("-----------------------------------")
