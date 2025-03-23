import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    def distance_field(self, img: torch.Tensor) -> torch.Tensor:
        """Computes the distance transform field efficiently on the GPU."""
        field = torch.zeros_like(img, device=img.device)
        for batch in range(img.shape[0]):
            fg_mask = img[batch] > 0.5
            if fg_mask.any():
                bg_mask = ~fg_mask
                fg_dist = torch.from_numpy(edt(fg_mask.cpu().numpy())).to(img.device)
                bg_dist = torch.from_numpy(edt(bg_mask.cpu().numpy())).to(img.device)
                field[batch] = fg_dist + bg_dist
        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.dim() in (4, 5), "Only 2D and 3D supported"
        assert pred.dim() == target.dim(), "Prediction and target must have the same dimension"

        pred_dt = self.distance_field(pred)
        target_dt = self.distance_field(target)

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        loss = (pred_error * distance).mean()
        return loss

class CMMLoss(nn.Module):
    def __init__(self, threshold=0.5, weight_ce=1.0, weight_dice=1.0, weight_hd=1.0, weight_cr=1.0):
        super(CMMLoss, self).__init__()
        self.threshold = threshold
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_hd = weight_hd
        self.weight_cr = weight_cr
        self.hausdorff_loss = HausdorffDTLoss()
    
    def count_connected_components(self, binary_image):
        """Counts the number of connected components in a binary image using BFS."""
        H, W = binary_image.shape
        visited = torch.zeros_like(binary_image, dtype=torch.bool)
        component_count = 0
        
        def bfs(x, y):
            queue = deque([(x, y)])
            visited[x, y] = True
            while queue:
                cx, cy = queue.popleft()
                for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1),
                               (cx+1, cy+1), (cx+1, cy-1), (cx-1, cy+1), (cx-1, cy-1)]:
                    if 0 <= nx < H and 0 <= ny < W and not visited[nx, ny] and binary_image[nx, ny] == 1:
                        visited[nx, ny] = True
                        queue.append((nx, ny))
        
        for i in range(H):
            for j in range(W):
                if binary_image[i, j] == 1 and not visited[i, j]:
                    bfs(i, j)
                    component_count += 1
        
        return component_count

    def connected_region_loss(self, predicted_segmentation):
        """Computes the connected region loss for a batch of segmentations."""
        batch_size = predicted_segmentation.size(0)
        total_loss = torch.zeros(batch_size, device=predicted_segmentation.device)
        
        for i in range(batch_size):
            binary_image = predicted_segmentation[i].squeeze().int()
            C = self.count_connected_components(binary_image)
            total_loss[i] = 1 - (1 / C) if C > 0 else 1
        
        return total_loss.mean()

    def dice_loss(self, predicted_segmentation, ground_truth):
        """Computes the Dice loss efficiently."""
        smooth = 1.0
        intersection = (predicted_segmentation * ground_truth).sum(dim=(1, 2, 3))
        total = predicted_segmentation.sum(dim=(1, 2, 3)) + ground_truth.sum(dim=(1, 2, 3))
        return (1 - (2.0 * intersection + smooth) / (total + smooth)).mean()

    def forward(self, predicted_segmentation, ground_truth):
        """Computes the combined loss efficiently."""
        ce_loss = F.binary_cross_entropy(predicted_segmentation, ground_truth.float(), reduction='mean')

        predicted_binary = (predicted_segmentation > self.threshold).int()
        
        dice_loss = self.dice_loss(predicted_binary, ground_truth)
        hd_loss = self.hausdorff_loss(predicted_binary, ground_truth)
        cr_loss = self.connected_region_loss(predicted_binary)
        
        total_loss = (self.weight_ce * ce_loss +
                      self.weight_dice * dice_loss +
                      self.weight_hd * hd_loss +
                      self.weight_cr * cr_loss)
        
        return total_loss

class CMMLossBase(nn.Module):
    def __init__(self, threshold=0.5, weight_ce=1.0, weight_dice=1.0, weight_hd=1.0, weight_cr=1.0):
        super(CMMLossBase, self).__init__()
        self.threshold = threshold
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.hausdorff_loss = HausdorffDTLoss()

    def dice_loss(self, predicted_segmentation, ground_truth):
        """Computes the Dice loss efficiently."""
        smooth = 1.0
        intersection = (predicted_segmentation * ground_truth).sum(dim=(1, 2, 3))
        total = predicted_segmentation.sum(dim=(1, 2, 3)) + ground_truth.sum(dim=(1, 2, 3))
        return (1 - (2.0 * intersection + smooth) / (total + smooth)).mean()

    def forward(self, predicted_segmentation, ground_truth):
        """Computes the combined loss efficiently."""
        ce_loss = F.binary_cross_entropy(predicted_segmentation, ground_truth.float(), reduction='mean')

        predicted_binary = (predicted_segmentation > self.threshold).int()
        
        dice_loss = self.dice_loss(predicted_binary, ground_truth)
        # print(ce_loss, dice_loss)
        
        total_loss = (self.weight_ce * ce_loss +
                      self.weight_dice * dice_loss)
        
        return total_loss

# Example usage
if __name__ == "__main__":
    predicted_segmentation = torch.rand(1, 1, 8, 8).cuda()
    ground_truth = torch.randint(0, 2, (1, 1, 8, 8), dtype=torch.float).cuda()
    
    cmm_loss_fn = CMMLoss().cuda()
    loss = cmm_loss_fn(predicted_segmentation, ground_truth)
    print(f"Total CMM Loss: {loss.item()}")
