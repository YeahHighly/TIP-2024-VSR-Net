import torch
import torch.nn as nn
# class SoftDiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(SoftDiceLoss, self).__init__()
 
#     def forward(self, logits, targets):
#         num = targets.size(0)
#         smooth = 1e-7

#         # probs = F.sigmoid(logits)
#         # probs = torch.sigmoid(logits)
#         m1 = logits.view(num, -1)
#         m2 = targets.view(num, -1)

#         intersection = (m1 * m2)
 
#         score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)

#         score = 1 - score.sum() / num
#         return score

class DiceLoss(nn.Module):
    def __init__(self):
	    super(DiceLoss, self).__init__()
        # self.num_classes = 1
 
    def	forward(self, input, target):
        N = target.size(0)
        smooth = 1
    
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
    
        intersection = input_flat * target_flat
    
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
    
        return loss

class VSegLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(VSegLoss, self).__init__()
        self.SoftDiceLoss = DiceLoss()
        # self.MSELoss = torch.nn.MSELoss(reduction='none')
        self.MSELoss = torch.nn.MSELoss()
 
    def forward(self, logits, targets):
        dice_loss = self.SoftDiceLoss(logits, targets)
        mse_loss = self.MSELoss(logits, targets)#.view(logits.shape[0], -1)


        # mse_loss, indices = torch.sort(mse_loss, descending=True)

        # mse_loss = mse_loss[:, :int(mse_loss.shape[1]*0.5)]
        # mse_loss = torch.mean(mse_loss)

        loss = dice_loss + mse_loss

        return loss