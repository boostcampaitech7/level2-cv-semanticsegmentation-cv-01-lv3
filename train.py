import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils.dataset import XRayDataset, CLASSES
from utils.trainer import train, set_seed
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.loss import CombinedBCEDiceLoss
import time
import torch

# Wandb import
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='X-Ray 이미지 세그멘테이션 학습')
    
    parser.add_argument('--image_root', type=str, default='./data/train/DCM',
                        help='학습 이미지가 있는 디렉토리 경로')
    parser.add_argument('--label_root', type=str, default='./data/train/outputs_json',
                        help='라벨 json 파일이 있는 디렉토리 경로')
    parser.add_argument('--model_name', type=str, default='FocalLoss',
                        help='모델 이름')
    parser.add_argument('--saved_dir', type=str, default='./checkpoints',
                        help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='총 에폭 수')
    parser.add_argument('--val_every', type=int, default=1,
                        help='검증 주기')
    
    # Wandb logging
    parser.add_argument('--wandb_project', type=str, default='UPerNet_Exp',
                        help='Wandb 프로젝트 이름')
    parser.add_argument('--wandb_entity', type=str, default='cv01-HandBone-seg',
                        help='Wandb 팀/조직 이름')
    parser.add_argument('--wandb_run_name', type=str, default='CombinedLoss_1024', help='encoder_name')

    return parser.parse_args()

def main():
    args = parse_args()
    
    current_time = time.strftime("%m-%d_%H-%M-%S")
    args.saved_dir = os.path.join(args.saved_dir, f"{current_time}_{args.model_name}")
    # Wandb initialize
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args)
    )

    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    print(f"Training Results will be saved in {args.saved_dir}!")
    
    # Set seed
    set_seed()
    
    # Dataset and DataLoader setup
    train_transform = A.Compose([
        A.Resize(1024, 1024)
    ])

    train_dataset = XRayDataset(args.image_root, args.label_root, is_train=True, transforms=train_transform)
    valid_dataset = XRayDataset(args.image_root, args.label_root, is_train=False, transforms=train_transform)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )
    
    # Model setup
    model = smp.UPerNet(
        encoder_name='tu-hrnet_w64', 
        in_channels=3, 
        classes=len(CLASSES)
    ).cuda()

    # Loss function and optimizer setup
    criterion = CombinedBCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,           # First restart epoch count
        T_mult=2,        # Increase period by T_mult for each restart
        eta_min=1e-6     # Minimum learning rate
    )

    # Set the number of accumulation steps
    accumulation_steps = 4  # Update every 4 steps

    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        for step, (images, targets) in enumerate(train_loader):
            images = images.cuda()
            targets = targets.cuda()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)  # Calculate loss

            # Normalize the loss by the number of accumulation steps
            loss = loss / accumulation_steps

            # Backward pass
            loss.backward()  # Accumulate gradients

            # Update weights every 'accumulation_steps'
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()  # Update weights
                optimizer.zero_grad()  # Reset gradients

            running_loss += loss.item()

        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Print or log the loss for the epoch
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Optionally, you can add validation logic here

    # Save the model after training
    torch.save(model.state_dict(), os.path.join(args.saved_dir, f"{args.model_name}.pth"))

if __name__ == '__main__':
    main()