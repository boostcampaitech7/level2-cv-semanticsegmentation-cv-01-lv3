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
    parser.add_argument('--model_name', type=str, default='hrnet_w64',
                        help='모델 이름')
    parser.add_argument('--saved_dir', type=str, default='./checkpoints',
                        help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='총 에폭 수')
    parser.add_argument('--val_every', type=int, default=5,
                        help='검증 주기')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient Accumulation Steps를 설정')
    # 실험 관리 용도
    parser.add_argument('--augmentation', type=str, default='', help='Checkpoint를 저장하는 디렉토리 이름 저장 용도')
    # Wandb logging
    parser.add_argument('--wandb_project', type=str, default='UPerNet_Exp_Augmentation',
                        help='Wandb 프로젝트 이름')
    parser.add_argument('--wandb_entity', type=str, default='cv01-HandBone-seg',
                        help='Wandb 팀/조직 이름')
    parser.add_argument('--wandb_run_name', type=str, default='Initial Test', help='WandB Run 이름')
    
    # Early stopping 관련 인자 수정
    parser.add_argument('--early_stopping', type=bool, default=True,
                      help='Enable early stopping (default: True)')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping patience (default: 5)')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    current_time = time.strftime("%m-%d_%H-%M-%S")
    # args.saved_dir = os.path.join(args.saved_dir, f"{current_time}_{args.model_name}")
    args.saved_dir = os.path.join(args.saved_dir, f"{current_time}_{args.augmentation}")
    # Wandb initalize
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
        A.Resize(1024,1024)
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
        encoder_weights='imagenet', 
        in_channels=3, 
        classes=len(CLASSES)
    ).cuda()

    # Loss function and optimizer setup
    criterion = CombinedBCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr= 0.001, pct_start=0.1, steps_per_epoch=len(train_loader)//args.batch_size, epochs=args.num_epochs, anneal_strategy='cos')
    # 학습 수행
    train(model, train_loader, valid_loader, criterion, optimizer, scheduler,
          args.num_epochs, args.val_every, args.saved_dir, args.model_name, wandb=wandb, accumulation_step=args.accumulation_steps)

if __name__ == '__main__':
    main()