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
import torch

# Wandb import(Feature:#3 Wandb logging, deamin, 2024.11.12)
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='X-Ray 이미지 세그멘테이션 학습')
    
    parser.add_argument('--image_root', type=str, default='./data/train/DCM',
                        help='학습 이미지가 있는 디렉토리 경로')
    parser.add_argument('--label_root', type=str, default='./data/train/outputs_json',
                        help='라벨 json 파일이 있는 디렉토리 경로')
    parser.add_argument('--model_name', type=str, default='UPerNet',
                        help='모델 이름')
    parser.add_argument('--saved_dir', type=str, default='./checkpoints',
                        help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='총 에폭 수')
    parser.add_argument('--val_every', type=int, default=5,
                        help='검증 주기')
    
    # Wandb logging
    parser.add_argument('--wandb_project', type=str, default='UPerNet_Experiments_alexseo',
                        help='Wandb 프로젝트 이름')
    parser.add_argument('--wandb_entity', type=str, default='cv01-HandBone-seg',
                        help='Wandb 팀/조직 이름')
    parser.add_argument('--wandb_run_name', type=str, default='UPerNet', help='WandB Run 이름')

    return parser.parse_args()

def main():
    transforms = [
                  A.Compose([A.CLAHE(p=1.),
        A.Resize(512,512)]),
        A.Compose([A.CLAHE(p=1., clip_limit=8.),
        A.Resize(512,512)]),
        A.Compose([A.CenterCrop(height=1024, width=1024),
    A.Resize(512, 512)]),
    A.Compose([
    A.Crop(x_min=1300-512, y_min=1000-512, x_max=1300+512, y_max=1000+512),  # 중심 (1000, 1300)에서 1024x1024 영역 자르기
    A.Resize(512, 512)
]), A.Compose([
    A.RandomCrop(width=512, height=512),  # 512x512 크기로 랜덤 크롭
    A.Resize(512, 512)  # 크롭 후 리사이즈
]),
A.Compose([A.RandomCrop(width=1024,height=1024)]), A.Compose([A.Resize(1024, 1024)])]
    val_transforms = [A.Compose([A.Resize(512,512), A.CLAHE(p=1.)]),A.Compose([A.Resize(512,512),A.CLAHE(p=1., clip_limit=8.)]),A.Compose([A.Resize(512,512)]),A.Compose([A.Resize(512,512)]),A.Compose([A.Resize(512,512)]), A.Compose([A.Resize(1024,1024)]),A.Compose([A.Resize(1024,1024)])]
    for i,(transform,v_transform) in enumerate(zip(transforms,val_transforms)):
        args = parse_args()
        current_time = time.strftime("%m-%d_%H-%M-%S")
        args.saved_dir = os.path.join(args.saved_dir, f"trans_{i+1}_{current_time}_{args.model_name}")
        # Wandb initalize
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.wandb_run_name}_trans_{i+1}",
            config=vars(args)
        )

        if not os.path.exists(args.saved_dir):
            os.makedirs(args.saved_dir)
        print(f"Training Results will be saved in {args.saved_dir}!")
        
        # 시드 고정
        set_seed()
        
        # 데이터셋 및 데이터로더 설정
        train_transform = transform

        v_transform = v_transform

        train_dataset = XRayDataset(args.image_root, args.label_root, is_train=True, transforms=train_transform)
        valid_dataset = XRayDataset(args.image_root, args.label_root, is_train=False, transforms=v_transform)

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=6,
            drop_last=True,
            pin_memory=False
        )
        
        valid_loader = DataLoader(
            dataset=valid_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            drop_last=False,
            pin_memory=False
        )
        
        # Torchvision 사용 시 주석 처리 해제
        #model = models.segmentation.fcn_resnet50(pretrained=True)
        #model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

        # 모델 smp로 설정 (모델 변경 시 수정 필요)
        model = smp.UPerNet(
            encoder_name='resnet50', 
            encoder_weights='imagenet', 
            in_channels=3, 
            classes=len(CLASSES)
            )
        
        # Loss function과 optimizer 설정
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
        
        # 학습 수행
        train(model, train_loader, valid_loader, criterion, optimizer, 
            args.num_epochs, args.val_every, args.saved_dir, args.model_name, wandb=wandb)
        wandb.finish()
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()