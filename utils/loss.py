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

class CombinedBCEDiceLossPosweight(nn.Module):
    def __init__(self, num_classes=29, bce_weight=0.5):
        """
        Combined Loss using BCEWithLogitsLoss and Dice Loss
        Args:
            num_classes: Number of classes in the segmentation task
            bce_weight: Weighting factor to balance BCE and Dice Loss
        """
        super(CombinedBCEDiceLossPosweight, self).__init__()

        # Class-wise positive weights (for BCE)
        pos_weight_values = [
            2.5491, 1.8302, 1.3358, 3.0197, 2.3485, 1.5581, 1.055, 2.8711,
            2.106, 1.4489, 1.1448, 2.8844, 2.2307, 1.6165, 1.4383, 3.2594,
            2.7788, 1.9714, 1.4232, 2.3718, 2.8579, 1.8734, 2.1939, 2.2624,
            2.4966, 2.6428, 3.1541, 0.3, 0.944
        ]
        
        # Ensure pos_weight is of correct size (num_classes,)
        pos_weight = torch.tensor(pos_weight_values[:num_classes], dtype=torch.float32)
        
        # Adjust pos_weight for broadcasting (size [num_classes, 1, 1])
        pos_weight = pos_weight.view(num_classes, 1, 1).cuda()

        # BCEWithLogitsLoss with pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        # Ensure inputs and targets are on the same device
        device = 'cuda'
        targets = targets.to(device)  # Move targets to the same device as inputs
        
        # Ensure targets have the same shape as inputs
        if targets.dim() == 3:  # If targets have shape [batch_size, height, width]
            targets = targets.unsqueeze(1)  # Add a channel dimension
            # Convert targets to one-hot encoding (if necessary)
            targets = F.one_hot(targets, num_classes=inputs.size(1))  # One-hot encoding
        
        # Ensure the shape is now [B, num_classes, H, W]
        bce = self.bce_loss(inputs, targets.float())  # Convert to float for BCE
        dice = self.dice_loss(inputs, targets.float())
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
