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
from timm.models.swin_transformer import SwinTransformer
from segmentation_models_pytorch.encoders._base import EncoderMixin
from typing import List

# Wandb import(Feature:#3 Wandb logging, deamin, 2024.11.12)
import wandb

# Custom SwinEncoder 정의
class SwinEncoder(torch.nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels: List[int] = [128, 256, 512, 1024]

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = 3

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3
        kwargs.pop('depth')
        print(kwargs)

        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        print(f"before: {x.shape}")
        outs = self.model(x)
        print(f"after: {outs.shape}") #여기까진 잘 나옴
        print(f"length of outs: {len(outs)}")
        return outs

    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict['model'], strict=False, **kwargs)

# Swin을 smp의 encoder로 사용할 수 있게 등록
def register_encoder():
    smp.encoders.encoders["swin_encoder"] = {
    "encoder": SwinEncoder, # encoder class here
    "pretrained_settings": { # pretrained 값 설정
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": { # 기본 파라미터
        "img_size": 512,
        'num_classes':len(CLASSES),
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        'num_heads': (4, 8, 16, 32),
        "window_size": 12,
        "drop_path_rate": 0.3,
    }
}
def parse_args():
    parser = argparse.ArgumentParser(description='X-Ray 이미지 세그멘테이션 학습')
    
    parser.add_argument('--image_root', type=str, default='./data/train/DCM',
                        help='학습 이미지가 있는 디렉토리 경로')
    parser.add_argument('--label_root', type=str, default='./data/train/outputs_json',
                        help='라벨 json 파일이 있는 디렉토리 경로')
    parser.add_argument('--model_name', type=str, default='UPerNet(SWIN)',
                        help='모델 이름')
    parser.add_argument('--saved_dir', type=str, default='./checkpoints',
                        help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='총 에폭 수')
    parser.add_argument('--val_every', type=int, default=5,
                        help='검증 주기')
    
    # Wandb logging
    parser.add_argument('--wandb_project', type=str, default='This will be Deleted',
                        help='Wandb 프로젝트 이름')
    parser.add_argument('--wandb_entity', type=str, default='cv01-HandBone-seg',
                        help='Wandb 팀/조직 이름')
    parser.add_argument('--wandb_run_name', type=str, default='swin', help='WandB Run 이름')

    return parser.parse_args()

def main():
    args = parse_args()
    
    current_time = time.strftime("%m-%d_%H-%M-%S")
    args.saved_dir = os.path.join(args.saved_dir, f"DELETE_{current_time}_{args.model_name}")
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
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    
    # Torchvision 사용 시 주석 처리 해제
    #model = models.segmentation.fcn_resnet50(pretrained=True)
    #model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    # model = smp.UPerNet(
    #     encoder_name= "swin_encoder",  # Swin Transformer 모델 사용
    #     encoder_depth= 4,  # Swin Transformer의 depth 설정
    #     encoder_weights = "imagenet",  # Swin 가중치 경로
    #     decoder_pyramid_channels = 512,  # 피라미드 채널 수
    #     decoder_segmentation_channels= 128,  # 세분화 채널 수
    #     in_channels=3,  # RGB 이미지 입력
    #     classes=len(CLASSES) ,  # 클래스 수
    # )
    # SMP UPerNet에 Swin Transformer Encoder 적용
    model = smp.UPerNet(
        encoder_name="swin_encoder",                          # Custom encoder 사용 시 None
        encoder_weights=None,                      # Custom encoder이므로 weights는 None
        in_channels=3,                             # 입력 채널 (RGB 이미지)
        classes=len(CLASSES)                       # 출력 클래스 수
    )
    # model = smp.UPerNet(
    #     encoder_name='tu-swin_large_patch4_window7_224', 
    #     encoder_weights='imagenet', 
    #     in_channels=3, 
    #     classes=len(CLASSES)
    #     )
    
    # Loss function과 optimizer 설정
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    # 학습 수행
    train(model, train_loader, valid_loader, criterion, optimizer, None, args.num_epochs, args.val_every, args.saved_dir, args.model_name, wandb=wandb)

if __name__ == '__main__':
    register_encoder()
    main()