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
import segmentation_models_pytorch as smp
# Wandb import(Feature:#3 Wandb logging, deamin, 2024.11.12)
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='X-Ray 이미지 세그멘테이션 학습')
    
    parser.add_argument('--image_root', type=str, default='./data/train/DCM',
                        help='학습 이미지가 있는 디렉토리 경로')
    parser.add_argument('--label_root', type=str, default='./data/train/outputs_json',
                        help='라벨 json 파일이 있는 디렉토리 경로')
    parser.add_argument('--model_name', type=str, default='Unetpp',
                        help='모델 이름')
    parser.add_argument('--saved_dir', type=str, default='./checkpoints',
                        help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=8, # 
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=3e-4, # 
                        help='학습률')
    parser.add_argument('--num_epochs', type=int, default=120,
                        help='총 에폭 수')
    parser.add_argument('--val_every', type=int, default=1,
                        help='검증 주기')
    
    # Wandb logging
    parser.add_argument('--wandb_project', type=str, default='Unetpp encoder',
                        help='Wandb 프로젝트 이름')
    parser.add_argument('--wandb_entity', type=str, default='cv01-HandBone-seg',
                        help='Wandb 팀/조직 이름')
    parser.add_argument('--wandb_run_name', type=str, default='Unetpp_efficientb0_3e4_adam_', help='WandB Run 이름')

    return parser.parse_args()

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
    
    # 모델 설정
    # model = models.segmentation.fcn_resnet50(pretrained=True)
    # model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    # model=smp.MAnet(
    #     encoder_name ="mit_b5",
    #     encoder_weights="imagenet",
    #     decoder_pab_channels=64,
    #     in_channels = 3,
    #     classes = 29
    # )

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0", 
        encoder_weights='imagenet', 
        in_channels=3, 
        classes=len(CLASSES)
    )

    # Loss function과 optimizer 설정
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
   
    # segmetataion pytorch

    # 학습 수행
    train(model, train_loader, valid_loader,criterion, optimizer, 
          args.num_epochs, args.val_every, args.saved_dir, args.model_name, wandb=wandb)

if __name__ == '__main__':
    main()