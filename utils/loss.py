import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice_score  # Return the Dice loss 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        focal_loss = -self.alpha * (1 - inputs_flat) ** self.gamma * targets_flat * torch.log(inputs_flat) - \
                     (1 - self.alpha) * inputs_flat ** self.gamma * (1 - targets_flat) * torch.log(1 - inputs_flat)
        return focal_loss.mean()

class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        """
        Combined Loss using BCEWithLogitsLoss and Dice Loss
        Args:
            bce_weight: Weighting factor to balance BCE and Dice Loss
        """
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice