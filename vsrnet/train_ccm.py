# encoding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from prettytable import PrettyTable
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score

import module
import dataloader  

# ------------------------ Argument Parser for Hyperparameters ------------------------
parser = argparse.ArgumentParser(description="Graph Edge Classification")
parser.add_argument("--module", type=str, default='ccm_plus', help="Graph classification backbone. 'ccm_plus' or 'ccm'") 
parser.add_argument("--dataset", type=str, default='drive', help="Graph dataset.")

parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and testing.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate model every N epochs.")
parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden channel size for GCN.")
parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device index (default: 0).")
parser.add_argument("--save_path", type=str, default="./checkpoints", help="Path to save model checkpoints.")

args = parser.parse_args()

# ------------------------ Environment Configuration ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.save_path = os.path.join(args.save_path, args.dataset, args.module)
os.makedirs(args.save_path, exist_ok=True)


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1):
        """
        Focal Loss with Label Smoothing for binary classification.

        Args:
            alpha (float): Balancing factor for positive class (default: 0.25).
            gamma (float): Focusing parameter (default: 2.0).
            smoothing (float): Label smoothing factor (default: 0.1).
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, probs, targets):
        """
        Compute Focal Loss with Label Smoothing for binary classification.

        Args:
            probs (Tensor): Model output after sigmoid (shape: [batch_size]).
            targets (Tensor): Ground truth labels (shape: [batch_size]).

        Returns:
            Tensor: Focal loss value.
        """
        # Apply label smoothing
        with torch.no_grad():
            targets = targets.float()
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing  # Smoothing applied

        # Compute focal weight
        focal_weight = (targets * (1 - probs) + (1 - targets) * probs) ** self.gamma

        # Compute BCE loss
        bce_loss = - (targets * torch.log(probs + 1e-8) + (1 - targets) * torch.log(1 - probs + 1e-8))

        # Compute focal loss
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean()


# ------------------------ Metric Calculation ------------------------
def compute_metrics(preds, targets):
    """
    Compute binary classification evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    """
    preds = (preds >= 0.5).cpu().numpy()  # Convert probabilities to binary predictions
    targets = targets.cpu().numpy()  # Convert targets to numpy

    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=1)
    recall = recall_score(targets, preds, zero_division=1)

    return acc, precision, recall


# ------------------------ Training Function ------------------------
def train():
    print("Initializing Training...")

    # Load dataset
    datafactory = dataloader.CCMDataFactory()
    train_loader = datafactory.get_dataset(args.dataset, batch_size=args.batch_size, train=True)
    test_loader = datafactory.get_dataset(args.dataset, batch_size=args.batch_size, train=False)

    # Initialize the model
    sample_data = next(iter(train_loader))  # Get a sample batch
    in_channels = sample_data[0].x.shape[1]  # Node feature size
    num_classes = 1 # Number of edge label classes
    
    ccmfactory = module.CCMFactory()
    model = ccmfactory.get_model(args.module, num_classes=num_classes)
    model = model.to(DEVICE)

    # Loss function & Optimizer
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Learning rate scheduler (Cosine Annealing)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            batch = [data.to(DEVICE) for data in batch]  # Move batch to GPU
            optimizer.zero_grad()
            # print(len(batch))
            total_loss = 0
            for data in batch:

                outputs = model(data)
                # print(outputs.shape, data.edge_label.shape, data.edge_label)
                loss = criterion(outputs, data.edge_label)
                loss.backward()
                total_loss += loss.item()
            
            optimizer.step()
            running_loss += total_loss / len(batch)

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # Training log
        print(f"[Epoch {epoch}/{args.epochs}] - Loss: {avg_loss:.4f} - Time: {time.time() - start_time:.2f}s")

        # Evaluate the model at specified intervals
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print("\nEvaluating Model...")
            acc_list, precision_list, recall_list = [], [], []

            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = [data.to(DEVICE) for data in batch]

                    for data in batch:
                        outputs = model(data)
                        acc, precision, recall = compute_metrics(outputs, data.edge_label)

                        acc_list.append(acc)
                        precision_list.append(precision)
                        recall_list.append(recall)

            # Compute average evaluation metrics
            mean_acc = np.mean(acc_list)
            mean_precision = np.mean(precision_list)
            mean_recall = np.mean(recall_list)

            # Display results in a formatted table
            table = PrettyTable(["Metric", "Value"])
            table.add_row(["Accuracy", f"{mean_acc:.4f}"])
            table.add_row(["Precision", f"{mean_precision:.4f}"])
            table.add_row(["Recall", f"{mean_recall:.4f}"])
            print(table)

            # Save the best model based on Accuracy
            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(model.state_dict(), os.path.join(args.save_path, "best.pth"))
                print("Best Model Updated (Saved as best.pth)")

        # Save the latest model snapshot
        torch.save(model.state_dict(), os.path.join(args.save_path, "latest.pth"))


if __name__ == "__main__":
    train()
