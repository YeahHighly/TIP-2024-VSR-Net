import os
import time
import torch
import numpy as np
import argparse
from prettytable import PrettyTable
import network
import dataloader
from metrics import compute_metrics, compute_metrics_simple

# ------------------------ Argument Parser for Hyperparameters ------------------------
parser = argparse.ArgumentParser(description="coarse_segmentation")
parser.add_argument("--netwrok", type=str, default='cenet', help="cenet, unet, csnet, skelcon")
parser.add_argument("--dataset", type=str, default='octa', help="drive, octa") 
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing.")
parser.add_argument("--cuda_device", type=str, default="1", help="CUDA device index (default: 0).")
parser.add_argument("--model_path", type=str, default="./checkpoints/", help="Path to saved model checkpoint.")
args = parser.parse_args()

# ------------------------ Environment Configuration ------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ Inference and Evaluation Function ------------------------
def evaluate_model():
    print("Initializing Evaluation...")

    # Load dataset
    factory = dataloader.DataFactory()
    test_loader = factory.get_dataset(args.dataset, batch_size=args.batch_size, train=False)

    # Initialize the model
    model_factory = network.ModelFactory()
    model = model_factory.get_model(args.network, num_classes=1, num_channels=3)

    model = model.to(DEVICE)

    # Load pre-trained model
    model.load_state_dict(torch.load(os.path.join(args.model_path, args.dataset, args.network, "best.pth")))
    model.eval()

    # Note: The original morphological evaluation difference metrics were calculated using MATLAB. 
    # The Python output results are for reference only.

    # Evaluation loop
    print("\nEvaluating Model...")
    
    model.eval()
    with torch.no_grad():
        # Initialize lists to store results for each batch
        all_metrics = {
            "PA": [],
            "Dice": [],
            "Jaccard": [],
            "VBN": [],
            "FD": [],
            "VT": [],
            "Beta_Error": [],
            "SMD": [],
            "ECE": []
        }

        # all_metrics = {
        #     "PA": [],
        #     "Dice": [],
        #     "Jaccard": [],
        # }

        for images, masks in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.cpu().squeeze(1).numpy()
            targets = masks.cpu().squeeze(1).numpy()
            preds = (preds > 0.5).astype(np.uint8)

            # Compute metrics for this batch
            # batch_metrics = compute_metrics_simple(preds, targets)
            batch_metrics = compute_metrics(preds, targets)

            # Append results to the lists
            for metric_name in all_metrics:
                all_metrics[metric_name].append(batch_metrics[metric_name])

            # break
        # Compute the average for each metric
        avg_metrics = {metric_name: np.mean(values) for metric_name, values in all_metrics.items()}

        # Display results in a horizontal table format
        table = PrettyTable()
        table.field_names = ["Metric"] + list(avg_metrics.keys())  # First row as metric names
        table.add_row(["Value"] + [f"{avg_metrics[metric]:.4f}" for metric in avg_metrics])  # Second row as values

        print(table)

if __name__ == "__main__":
    evaluate_model()
