import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import models
from utils.dataset import XRayDataset, CLASSES
from utils.trainer import train, set_seed
import time

# Wandb import(Feature:#3 Wandb logging, deamin, 2024.11.12)
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='X-Ray 이미지 세그멘테이션 학습')
    
    parser.add_argument('--image_root', type=str, default='./data/train/DCM',
                        help='학습 이미지가 있는 디렉토리 경로')
    parser.add_argument('--label_root', type=str, default='./data/train/outputs_json',
                        help='라벨 json 파일이 있는 디렉토리 경로')
    parser.add_argument('--model_name', type=str, default='fcn_resnet50',
                        help='모델 이름')
    parser.add_argument('--saved_dir', type=str, default='./checkpoints',
                        help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='총 에폭 수')
    parser.add_argument('--val_every', type=int, default=1,
                        help='검증 주기')
    
    # Wandb logging
    parser.add_argument('--wandb_project', type=str, default='FCN_baseline',
                        help='Wandb 프로젝트 이름')
    parser.add_argument('--wandb_entity', type=str, default='cv01-HandBone-seg',
                        help='Wandb 팀/조직 이름')
    parser.add_argument('--wandb_run_name', type=str, default='', help='WandB Run 이름')
    
    # Early stopping 관련 인자 수정
    parser.add_argument('--early_stopping', type=bool, default=True,
                      help='Enable early stopping (default: True)')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping patience (default: 5)')
    
    args = parser.parse_args()
    return args

# Hook import 추가
from utils.hook import FeatureExtractor

# DeepLabV3+ import
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.optim.lr_scheduler import _LRScheduler

# Custom Polynomial LR Scheduler with Warmup
class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, use_warmup=True, warmup_epochs=5, power=0.9, last_epoch=-1, num_epochs=100):
        self.total_iters = total_iters
        self.use_warmup = use_warmup
        self.warmup_iters = warmup_epochs * (total_iters // num_epochs) if use_warmup else 0
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.use_warmup and self.last_epoch < self.warmup_iters:
            # Warm-up 기간 동안 선형적으로 학습률 증가
            return [base_lr * (self.last_epoch / self.warmup_iters)
                    for base_lr in self.base_lrs]
        else:
            # Warm-up 이후 또는 Warm-up 미사용시 polynomial decay
            curr_iter = self.last_epoch - self.warmup_iters if self.use_warmup else self.last_epoch
            total_iter = self.total_iters - self.warmup_iters if self.use_warmup else self.total_iters
            return [base_lr * (1 - curr_iter/total_iter) ** self.power
                    for base_lr in self.base_lrs]

# Custom Loss for DeepLab v3+
class DeepLabLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super().__init__()
        self.main_loss = nn.BCEWithLogitsLoss()
        self.aux_loss = nn.BCEWithLogitsLoss()
        self.aux_weight = aux_weight
        
    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            # Interpolate aux_out to match target size
            aux_out = F.interpolate(
                aux_out.unsqueeze(-1).unsqueeze(-1),
                size=(512, 512),
                mode='bilinear',
                align_corners=True
            )
            loss = self.main_loss(main_out, targets) + self.aux_weight * self.aux_loss(aux_out, targets)
        else:
            loss = self.main_loss(outputs, targets)
        return loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # logits를 확률로 변환
        
        # 배치 차원을 따라 평균 계산
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedDeepLabLoss(nn.Module):
    def __init__(self, aux_weight=0.4, dice_weight=0.5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.aux_weight = aux_weight
        self.dice_weight = dice_weight
        
    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            
            # aux_out의 shape 확인 및 처리
            if aux_out.dim() == 2:  # (B, C) 형태인 경우
                aux_out = aux_out.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)로 변환
            
            # Interpolate aux_out to match target size
            aux_out = F.interpolate(
                aux_out,
                size=(512, 512),
                mode='bilinear',
                align_corners=True
            )
            
            # 주 출력에 대한 손실 계산
            main_bce = self.bce_loss(main_out, targets)
            main_dice = self.dice_loss(main_out, targets)
            main_loss = (1 - self.dice_weight) * main_bce + self.dice_weight * main_dice
            
            # 보조 출력에 대한 손실 계산
            aux_bce = self.bce_loss(aux_out, targets)
            aux_dice = self.dice_loss(aux_out, targets)
            aux_loss = (1 - self.dice_weight) * aux_bce + self.dice_weight * aux_dice
            
            loss = main_loss + self.aux_weight * aux_loss
        else:
            bce = self.bce_loss(outputs, targets)
            dice = self.dice_loss(outputs, targets)
            loss = (1 - self.dice_weight) * bce + self.dice_weight * dice
            
        return loss

def main():
    args = parse_args()
    
    current_time = time.strftime("%m-%d_%H-%M-%S")
    args.saved_dir = os.path.join(args.saved_dir, f"{current_time}_{args.model_name}")
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
    
    # 시드 고정
    set_seed()
    
    # 데이터셋 및 데이터로더 설정
    train_transform = A.Compose([
        A.Resize(512,512)
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
    
    # Torchvision 사용 시 주석 처리 해제
    #model = models.segmentation.fcn_resnet50(pretrained=True)
    #model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

    # # 모델 smp로 설정 (모델 변경 시 수정 필요)
    # model = smp.UPerNet(
    #     encoder_name='efficientnet-b0', 
    #     encoder_weights='imagenet', 
    #     in_channels=3, 
    #     classes=len(CLASSES)
    #     )
    import segmentation_models_pytorch as smp
    # # DeepLabV3+ 모델 설정
    import ssl
    import urllib.request

    # SSL 인증서 검증 비활성화
    #ssl._create_default_https_context = ssl._create_unverified_context

    # aux_params 수정
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(CLASSES),
        aux_params={
            "classes": len(CLASSES),
            "dropout": 0.5
        },
        decoder_channels=256,
        encoder_output_stride=8,
        decoder_atrous_rates=(4, 8, 12)
    )
    
    criterion = CombinedDeepLabLoss(aux_weight=0.4, dice_weight=0.5) 
    #optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=args.lr,
    #     weight_decay=0.01,
    #     betas=(0.9, 0.999)
    # )
    # AdamW optimizer 설정
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler 설정
    total_iters = args.num_epochs * len(train_loader)
    
    scheduler = PolynomialLR(
        optimizer,
        total_iters=total_iters,
        power=args.power,
        warmup_epochs=5,
        num_epochs=args.num_epochs,
        use_warmup=False
    )
    # # Loss function과 optimizer 설정
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    # 학습 수행
    train(model, train_loader, valid_loader, criterion, optimizer, 
            args.num_epochs, args.val_every, args.saved_dir, args.model_name, 
            wandb=wandb, accumulation_steps=args.accumulation_steps,
            scheduler=scheduler)
    
if __name__ == '__main__':
    main()