# encoding: utf-8
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import jaccard_score, f1_score

import network
import dataloader
from loss import VSegLoss

# ------------------------ Argument Parser for Hyperparameters ------------------------
parser = argparse.ArgumentParser(description="coarse_segmentation") 
parser.add_argument("--netwrok", type=str, default='cenet', help="cenet, unet, csnet, skelcon")
parser.add_argument("--dataset", type=str, default='octa', help="drive, octa") 

parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and testing.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate.")
parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate model every N epochs.")
parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device index (default: 0).")
parser.add_argument("--save_path", type=str, default="./checkpoints", help="Path to save model checkpoints.")

args = parser.parse_args()

# ------------------------ Environment Configuration ------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.save_path = os.path.join(args.save_path, args.dataset, args.netwrok)
os.makedirs(args.save_path, exist_ok=True)


# ------------------------ Metric Calculation ------------------------
def compute_metrics(preds, targets):
    """
    Compute segmentation evaluation metrics:
    - Pixel Accuracy (PA)
    - Dice Score (F1 Score)
    - Jaccard Index (IoU)
    """
    preds = (preds > 0.5).astype(np.uint8).flatten()
    targets = targets.astype(np.uint8).flatten()

    pa = (preds == targets).sum() / len(targets)  # Pixel Accuracy
    dice = f1_score(targets, preds, zero_division=1)  # Dice (F1 Score)
    jaccard = jaccard_score(targets, preds, zero_division=1)  # Jaccard Index (IoU)

    return pa, dice, jaccard


# ------------------------ Training Function ------------------------
def train():
    print("Initializing Training...")

    # Load dataset
    factory = dataloader.DataFactory()
    train_loader = factory.get_dataset(args.dataset, batch_size=args.batch_size, train=True)
    test_loader = factory.get_dataset(args.dataset, batch_size=args.batch_size, train=False)

    # Initialize the model
    model_factory = network.ModelFactory()
    model = model_factory.get_model(args.netwrok, num_classes=1, num_channels=3)
    model = model.to(DEVICE)

    # Loss function & Optimizer
    criterion = VSegLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler (Warm-up & Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training loop
    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # Training log
        print(f"[Epoch {epoch}/{args.epochs}] - Loss: {avg_loss:.4f} - Time: {time.time() - start_time:.2f}s")

        # Evaluate the model at specified intervals
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print("\nEvaluating Model...")
            pa_list, dice_list, jaccard_list = [], [], []

            model.eval()
            with torch.no_grad():
                for images, masks in test_loader:
                    images = images.to(DEVICE)
                    outputs = model(images)
                    preds = outputs.cpu().numpy()
                    targets = masks.cpu().numpy()

                    pa, dice, jaccard = compute_metrics(preds, targets)
                    pa_list.append(pa)
                    dice_list.append(dice)
                    jaccard_list.append(jaccard)

            # Compute average evaluation metrics
            mean_pa = np.mean(pa_list)
            mean_dice = np.mean(dice_list)
            mean_jaccard = np.mean(jaccard_list)

            # Display results in a formatted table
            table = PrettyTable(["Metric", "Value"])
            table.add_row(["Pixel Accuracy (PA)", f"{mean_pa:.4f}"])
            table.add_row(["Dice Score", f"{mean_dice:.4f}"])
            table.add_row(["Jaccard Index (IoU)", f"{mean_jaccard:.4f}"])
            print(table)

            # Save the best model based on Dice Score
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(args.save_path, "best.pth"))
                print("Best Model Updated (Saved as best.pth)")

        # Save the latest model snapshot
        torch.save(model.state_dict(), os.path.join(args.save_path, "latest.pth"))


if __name__ == "__main__":
    train()
