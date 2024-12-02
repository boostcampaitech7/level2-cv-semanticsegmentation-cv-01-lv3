import os
import albumentations as A
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import XRayDataset, CLASSES, SubsetBatchSampler
from utils.trainer import train, set_seed

from torch.nn import functional as F
# torch import
import torch 
# Wandb import(Feature:#3 Wandb logging, deamin, 2024.11.12)
import wandb

# Hook import 추가
from utils.hook import FeatureExtractor

# DeepLabV3+ import
import torch.nn as nn
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
                size=(1024, 1024),
                mode='bilinear',
                align_corners=True
            )
            loss = self.main_loss(main_out, targets) + self.aux_weight * self.aux_loss(aux_out, targets)
        else:
            loss = self.main_loss(outputs, targets)
        return loss

# Dice Loss
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

# class CombinedDeepLabLoss(nn.Module):
#     def __init__(self, aux_weight=0.4, dice_weight=0.5):
#         super().__init__()
#         self.bce_loss = nn.BCEWithLogitsLoss()
#         self.dice_loss = DiceLoss()
#         self.aux_weight = aux_weight
#         self.dice_weight = dice_weight
        
#     def forward(self, outputs, targets):
#         if isinstance(outputs, tuple):
#             main_out, aux_out = outputs
            
#             # aux_out의 shape 확인 및 처리
#             if aux_out.dim() == 2:  # (B, C) 형태인 경우
#                 aux_out = aux_out.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)로 변환
            
#             # Interpolate aux_out to match target size
#             aux_out = F.interpolate(
#                 aux_out,
#                 size=(1024, 1024),
#                 mode='bilinear',
#                 align_corners=True,
#                 antialias=True
#             )
            
#             # 주 출력에 대한 손실 계산
#             main_bce = self.bce_loss(main_out, targets)
#             main_dice = self.dice_loss(main_out, targets)
#             main_loss = (1 - self.dice_weight) * main_bce + self.dice_weight * main_dice
            
#             # 보조 출력에 대한 손실 계산
#             aux_bce = self.bce_loss(aux_out, targets)
#             aux_dice = self.dice_loss(aux_out, targets)
#             aux_loss = (1 - self.dice_weight) * aux_bce + self.dice_weight * aux_dice
            
#             loss = main_loss + self.aux_weight * aux_loss
#         else:
#             bce = self.bce_loss(outputs, targets)
#             dice = self.dice_loss(outputs, targets)
#             loss = (1 - self.dice_weight) * bce + self.dice_weight * dice
            
#         return loss
class CombinedDeepLabLoss(nn.Module):
    def __init__(self, num_classes=29, aux_weight=0.4, dice_weight=0.5):
        super().__init__()
        
        # Positive weights 설정
        pos_weight_values = [1.75998513, 1.51706427, 1.35000338, 1.91900385, 1.69220112, 1.42511996, 
                             1.25511928, 1.86879097, 1.61025884, 1.38822059, 1.28546327, 1.87328513,
                             1.65239576, 1.44485369, 1.38463878, 2., 1.83760222, 1.56477664,
                             1.37953639, 1.70007434, 1.86433061, 1.53166182, 1.6399608, 1.66310739,
                             1.74224505, 1.79164696, 1.96441846, 1., 1.21761168]

        pos_weight = torch.tensor(pos_weight_values[:num_classes], dtype=torch.float32)
        pos_weight = pos_weight.view(num_classes, 1, 1).cuda()
        
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss()
        self.aux_weight = aux_weight
        self.dice_weight = dice_weight
        
    def forward(self, outputs, targets):
        # 타겟을 cuda로 이동
        device = 'cuda'
        targets = targets.to(device)
        
        # 타겟 shape 처리
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
            targets = F.one_hot(targets, num_classes=outputs[0].size(1) if isinstance(outputs, tuple) else outputs.size(1))
            targets = targets.permute(0, 3, 1, 2).float()  # [B, C, H, W] 형태로 변환
        
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            
            # aux_out의 shape 처리
            if aux_out.dim() == 2:
                aux_out = aux_out.unsqueeze(-1).unsqueeze(-1)
            
            # aux_out을 타겟 크기로 보간
            aux_out = F.interpolate(
                aux_out,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=True,
                antialias=True
            )
            
            # 주 출력 손실 계산
            main_bce = self.bce_loss(main_out, targets)
            main_dice = self.dice_loss(main_out, targets)
            main_loss = (1 - self.dice_weight) * main_bce + self.dice_weight * main_dice
            
            # 보조 출력 손실 계산
            aux_bce = self.bce_loss(aux_out, targets)
            aux_dice = self.dice_loss(aux_out, targets)
            aux_loss = (1 - self.dice_weight) * aux_bce + self.dice_weight * aux_dice
            
            loss = main_loss + self.aux_weight * aux_loss
        else:
            bce = self.bce_loss(outputs, targets)
            dice = self.dice_loss(outputs, targets)
            loss = (1 - self.dice_weight) * bce + self.dice_weight * dice
            
        return loss
    
def parse_args():
    parser = argparse.ArgumentParser(description='X-Ray 이미지 세그멘테이션 학습')
    
    parser.add_argument('--image_root', type=str, default='./data/train/DCM',
                        help='학습 이미지가 있는 디렉토리 경로')
    parser.add_argument('--label_root', type=str, default='./data/train/outputs_json',
                        help='라벨 json 파일이 있는 디렉토리 경로')
    parser.add_argument('--model_name', type=str, default='1024_lr1e3_adamW_Fold3',
                        help='모델 이름')
    parser.add_argument('--saved_dir', type=str, default='./checkpoints',
                        help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='학습률')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='총 에폭 수')
    parser.add_argument('--val_every', type=int, default=1,
                        help='검증 주기')
    # poly learning rate 관련 인자 추가
    parser.add_argument('--power', type=float, default=0.9,
                        help='Poly learning rate power')
    
    # Wandb logging
    parser.add_argument('--wandb_project', type=str, default='DeepLabV3Plus_enhance_deamin',
                        help='Wandb 프로젝트 이름')
    parser.add_argument('--wandb_entity', type=str, default='cv01-HandBone-seg',
                        help='Wandb 팀/조직 이름')

    # gradient accumulation
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Hook 관련 arguments 추가
    parser.add_argument('--use_hook', default=True,
                        help='Feature visualization hook 사용 여부')
#    efficientnet-b7
    parser.add_argument('--target_layers', nargs='+',
                        default= [
                            'encoder.layer1',
                            'encoder.layer2',
                            'encoder.layer3',
                            'encoder.layer4',
                            'decoder.aspp.0.convs.0',
                            'decoder.aspp.0.convs.1',
                            'decoder.aspp.0.convs.2',
                            'decoder.aspp.0.convs.3',
                            'decoder.aspp.0.project',
                            'decoder.block1',
                            'decoder.block2',
                            'segmentation_head.2'
                        ],
                        help='Feature를 시각화할 target layers')

    # parser.add_argument('--target_layers', nargs='+', 
    #             default= [
    #                 # Encoder (MiT-B5) 계층적 특징 추출
    #                 'encoder.patch_embed1',          # 초기 패치 임베딩
    #                 'encoder.block1',               # Stage 1 특징
    #                 'encoder.patch_embed2',         # Stage 1->2 패치 병합
    #                 'encoder.block2',               # Stage 2 특징
    #                 'encoder.patch_embed3',         # Stage 2->3 패치 병합
    #                 'encoder.block3',               # Stage 3 특징
    #                 'encoder.patch_embed4',         # Stage 3->4 패치 병합
    #                 'encoder.block4',               # Stage 4 특징 (가장 추상화된 특징)
                    
    #                 # DeepLabV3+ Decoder
    #                 'decoder.aspp.pooling',         # ASPP 글로벌 컨텍스트
    #                 'decoder.aspp.convs.0',         # ASPP rate=1
    #                 'decoder.aspp.convs.1',         # ASPP rate=12
    #                 'decoder.aspp.convs.2',         # ASPP rate=24
    #                 'decoder.aspp.convs.3',         # ASPP rate=36
    #                 'decoder.aspp.project',         # ASPP 특징 융합
                    
    #                 # 최종 세그멘테이션 헤드
    #                 'segmentation_head.0',          # 최종 예측층
    #             ],
    #             help='Feature를 시각화할 target layers')
    # # efficientnet-b1
    # parser.add_argument('--target_layers', nargs='+', 
    #                 default= [
    #                     # Encoder (주요 특징 추출 단계)
    #                     'encoder.blocks.1.0',  # 초기 특징 추출
    #                     'encoder.blocks.2.0',  # 중간 레벨 특징
    #                     'encoder.blocks.4.0',  # 고수준 특징
    #                     'encoder.blocks.6.0',  # 가장 추상화된 특징
                        
    #                     # Decoder (ASPP와 디코더 부분)
    #                     'decoder.aspp.pooling',  # 글로벌 컨텍스트
    #                     'decoder.aspp.convs.0',  # ASPP 브랜치
    #                     'decoder.aspp.project',  # ASPP 출력
                        
    #                     # 최종 세그멘테이션 헤드
    #                     'segmentation_head.0',  # 최종 예측 직전
    #                 ],
    #                 help='Feature를 시각화할 target layers')

    # # efficientnet-b6
    # parser.add_argument('--target_layers', nargs='+', 
    #                 default= [
    #                     # Encoder의 주요 블록들
    #                     'encoder.blocks.1.0',  # 초기 특징 (32x32)
    #                     'encoder.blocks.2.1',  # 저수준 특징 (16x16)
    #                     'encoder.blocks.3.2',  # 중간 수준 특징
    #                     'encoder.blocks.4.3',  # 중-고수준 특징
    #                     'encoder.blocks.5.4',  # 고수준 특징
    #                     'encoder.blocks.6.5',  # 매우 추상화된 특징
    #                     'encoder.blocks.7.2',  # 가장 깊은 특징
                        
    #                     # Decoder와 ASPP 부분
    #                     'decoder.aspp.pooling',  # 글로벌 컨텍스트
    #                     'decoder.aspp.convs.0',  # ASPP 첫 번째 브랜치
    #                     'decoder.aspp.project',  # ASPP 최종 출력
                        
    #                     # 최종 세그멘테이션 부분
    #                     'segmentation_head.0'  # 최종 예측층
    #                 ],
    #                 help='Feature를 시각화할 target layers')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Wandb initalize
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.model_name,
        config=vars(args)
    )

    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    
    # 시드 고정
    set_seed()
    
    # import albumentations as A
    # import cv2
    # import numpy as np
    # class CustomPreprocessing(A.ImageOnlyTransform):
    #     def __init__(self, always_apply=False, p=1.0):
    #         super().__init__(always_apply, p)
            
    #     def apply(self, image, **params):
    #         # cv2 처리를 위해 uint8로 변환
    #         if image.dtype != np.uint8:
    #             image = (image * 255).astype(np.uint8)
                
    #         # grayscale 변환
    #         if len(image.shape) == 3:
    #             gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #         else:
    #             gray = image
                
    #         # Laplacian 적용
    #         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    #         laplacian = cv2.convertScaleAbs(laplacian)
            
    #         # 3채널로 변환하여 addWeighted 적용
    #         if len(image.shape) == 3:
    #             laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            
    #         enhanced = cv2.addWeighted(image.astype(np.float32), 1.0,
    #                                 laplacian.astype(np.float32), 0.2, 0)
            
    #         # CLAHE 적용
    #         enhanced_uint8 = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
    #         if len(enhanced_uint8.shape) == 3:
    #             # RGB 이미지인 경우 각 채널에 CLAHE 적용
    #             enhanced = np.zeros_like(enhanced_uint8)
    #             for i in range(3):
    #                 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #                 enhanced[:,:,i] = clahe.apply(enhanced_uint8[:,:,i])
    #         else:
    #             # 그레이스케일 이미지인 경우
    #             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #             enhanced = clahe.apply(enhanced_uint8)
                
    #         return enhanced
                
            
    # 데이터셋 transform 수정 - crop size를 513x513으로 변경
    train_transform = A.Compose([
        A.CLAHE(clip_limit=3.0, tile_grid_size=(3, 3), p=1.0),
        # A.OneOf([
        #     A.RandomCrop(768, 768),
        #     A.RandomCrop(512, 512),
        #     A.RandomCrop(1280, 1280),
        # ], p=1.0),        
        A.Resize(1024, 1024),  # 최종적으로 512x512로 리사이즈
        A.HorizontalFlip(p=0.5),
    ])

    valid_transform = A.Compose([
        #CustomPreprocessing(p=1),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(3, 3), p=1.0),
        A.Resize(1024, 1024),  # validation은 resize만 적용
        A.HorizontalFlip(p=0.5),
    ])
    
    train_dataset = XRayDataset(args.image_root, args.label_root, is_train=True, transforms=train_transform)
    valid_dataset = XRayDataset(args.image_root, args.label_root, is_train=False, transforms=valid_transform)
    
    # # BatchSampler 설정
    # train_batch_sampler = SubsetBatchSampler(
    #     data_source=train_dataset,
    #     batch_size=args.batch_size,        # 배치 크기
    #     subset_size=200,     # 한 번에 메모리에 로드할 데이터 수
    #     shuffle=True
    # )

    # valid_batch_sampler = SubsetBatchSampler(
    #     data_source=valid_dataset,
    #     batch_size=args.batch_size,        # 배치 크기
    #     subset_size=200,     # 한 번에 메모리에 로드할 데이터 수
    #     shuffle=False
    # )

    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_sampler=train_batch_sampler,
    #     num_workers=8,
    #     pin_memory=False
    # )

    # valid_loader = DataLoader(
    #     dataset=valid_dataset,
    #     batch_sampler=valid_batch_sampler,
    #     num_workers=8,
    #     pin_memory=False
    # )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=False
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=False
    )

    # # 3. 메모리 최적화 설정 추가
    # torch.cuda.empty_cache()  # 캐시 메모리 정리
    # backends.cudnn.benchmark = True  # 성능 최적화

    # 모델 설정
    # model = models.segmentation.fcn_resnet50(pretrained=True)    
    # model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

    # model = torch.load('./checkpoints/adam_30_fcn_resnet50.pt')
    # model = models.segmentation.deeplabv3plus_xception(pretrained=True)
    import segmentation_models_pytorch as smp
    # # DeepLabV3+ 모델 설정

    # SSL 인증서 검증 비활성화
    #ssl._create_default_https_context = ssl._create_unverified_context
    #from timm.models.swin_transformer import SwinTransformer
    # aux_params 수정
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b7",
        encoder_weights=None,
        in_channels=3,
        classes=len(CLASSES),
        aux_params={
            "classes": len(CLASSES),
            "dropout": 0.5
        },
        decoder_channels=256,
        encoder_output_stride=16,
        decoder_atrous_rates=(4, 8, 12),
        decoder_aspp_separable=True
    )

    # 저장된 모델 가중치 로드
    checkpoint_path = "/data/ephemeral/home/deamin/backup/level2-cv-semanticsegmentation-cv-01-lv3/wandb/run-20241128_063143-fold3/files/checkpoints/1024_lr1e3_adamW_Fold3.pt"  # 실제 저장된 모델 경로로 수정
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Model file not found at {checkpoint_path}")

    # # 디버깅을 위한 코드 추가
    # try:
    #     print("Creating Swin Transformer model...")
    #     import timm
        
    #     # 사용 가능한 모델 리스트 확인
    #     print("\nAvailable Swin models in timm:")
    #     available_models = timm.list_models('swin*')
    #     print(available_models)
        
    #     # 모델 생성 시도
    #     print("\nTrying to create DeepLabV3Plus with Swin...")
    #     model = smp.DeepLabV3Plus(
    #         encoder_name="swin_base_patch4_window7_224",
    #         encoder_weights="imagenet",
    #         in_channels=3,
    #         classes=len(CLASSES),
    #         aux_params={
    #             "classes": len(CLASSES),
    #             "dropout": 0.5
    #         },
    #         decoder_channels=256,
    #         encoder_output_stride=16,
    #         decoder_atrous_rates=(4, 8, 12)
    #     )
    #     print("Model created successfully!")
        
    #     # 모델 구조 출력
    #     print("\nModel structure:")
    #     print(model)
        
    # except Exception as e:
    #     print("\nError occurred during model creation:")
    #     print(f"Error type: {type(e).__name__}")
    #     print(f"Error message: {str(e)}")
    #     import traceback
    #     print("\nFull traceback:")
    #     traceback.print_exc()
    #     raise e
    
    # Beit-3 모델 설정
    #from transformers import Beit3Config
    # model = Beit3ForSemanticSegmentation.from_pretrained(
    #     "microsoft/beit-3-large",
    #     num_labels=len(CLASSES),
    #     ignore_mismatched_sizes=True,
    #     id2label={str(i): class_name for i, class_name in enumerate(CLASSES)},
    #     label2id={class_name: str(i) for i, class_name in enumerate(CLASSES)}
    # )
    # model.config.image_size = 512


    # from mmseg.apis import init_segmentor

    # # 설정 파일
    # config_file = 'configs/deeplabv3plus/deeplabv3plus_swin-tiny_hand.py'

    # # 설정 파일 예시
    # model = dict(
    #     type='DeepLabV3Plus',
    #     backbone=dict(
    #         type='SwinTransformer',
    #         embed_dims=96,
    #         depths=[2, 2, 6, 2],
    #         num_heads=[3, 6, 12, 24],
    #         window_size=7,
    #         pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
    #     ),
    #     decode_head=dict(
    #         type='DeepLabV3PlusHead',
    #         in_channels=768,
    #         in_index=3,
    #         channels=256,
    #         dilations=(1, 12, 24, 36),
    #         c1_in_channels=96,
    #         c1_channels=48,
    #         dropout_ratio=0.1,
    #         num_classes=num_classes,  # 본인의 클래스 수로 설정
    #         norm_cfg=dict(type='SyncBN', requires_grad=True),
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    # )
    # Loss function과 optimizer 설정
    # criterion = DeepLabLoss(aux_weight=0.4) 
    # criterion = nn.BCEWithLogitsLoss()
    # main 함수에서 criterion 정의할 때
    criterion = CombinedDeepLabLoss(aux_weight=0.4, dice_weight=0.5) 
    # criterion = CustomDiceFocalLoss()
    #criterion = CustomSegmentationLoss(alpha=0.3, beta=0.3, gamma=0.4)
    # criterion = EfficientBoneLoss(
    #     focal_gamma=2.5,
    #     tversky_alpha=0.8
    # )
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

    # from transformers import get_cosine_schedule_with_warmup

    
    # # 스케줄러 설정
    # steps_per_epoch = len(train_loader)
    # num_training_steps = args.num_epochs * steps_per_epoch
    # num_warmup_steps = 5 * steps_per_epoch  # 5 epoch 동안 워밍업

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps
    # )

    total_iters = args.num_epochs * len(train_loader)
    scheduler = PolynomialLR(
        optimizer,
        total_iters=total_iters,
        power=args.power,
        warmup_epochs=5,
        num_epochs=args.num_epochs,
        use_warmup=False
    )
    
    # Hook 설정 (선택적)
    # Training
    if args.use_hook:
        with FeatureExtractor(model, target_layers=args.target_layers) as feature_extractor:
            train(model, train_loader, valid_loader, criterion, optimizer, 
                  args.num_epochs, args.val_every, args.saved_dir, args.model_name, 
                  wandb=wandb, accumulation_steps=args.accumulation_steps,
                  scheduler=scheduler)
    else:
        train(model, train_loader, valid_loader, criterion, optimizer, 
              args.num_epochs, args.val_every, args.saved_dir, args.model_name, 
              wandb=wandb, accumulation_steps=args.accumulation_steps,
              scheduler=scheduler)
        
if __name__ == '__main__':
    main()
    
# from mmseg.models import build_segmentor
# from mmengine.config import Config  # 새로운 import 방식
# from mmseg.apis import train_segmentor
# from mmcv.runner import build_optimizer

# def main():
#     args = parse_args()
    
#     # Wandb initalize
#     wandb.init(
#         project=args.wandb_project,
#         entity=args.wandb_entity,
#         name=args.model_name,
#         config=vars(args)
#     )

#     # MMSegmentation config 로드
#     cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus-swin-tiny.py')
    
#     # config 수정
#     cfg.optimizer = dict(
#         type='AdamW',
#         lr=args.lr,
#         weight_decay=0.01,
#         betas=(0.9, 0.999),
#         paramwise_cfg=dict(
#             custom_keys={
#                 'absolute_pos_embed': dict(decay_mult=0.),
#                 'relative_position_bias_table': dict(decay_mult=0.),
#                 'norm': dict(decay_mult=0.)
#             }))
    
#     cfg.lr_config = dict(
#         policy='poly',
#         warmup='linear',
#         warmup_iters=1500,
#         warmup_ratio=1e-6,
#         power=args.power,
#         min_lr=0.0,
#         by_epoch=False
#     )
    
#     # 데이터셋 설정
#     cfg.data.train.data_root = args.image_root
#     cfg.data.train.ann_dir = args.label_root
#     cfg.data.val.data_root = args.image_root
#     cfg.data.val.ann_dir = args.label_root
    
#     # 모델 생성
#     model = build_segmentor(cfg.model)
    
#     # Hook 설정을 위한 wrapper 클래스
#     class MMSegHookWrapper(nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model
            
#         def forward(self, x):
#             return self.model.forward_train(x, None)  # MMSeg 모델 출력 형식 변환
    
#     model_with_hook = MMSegHookWrapper(model)
    
#     # Loss function과 optimizer 설정
#     criterion = DeepLabLoss(aux_weight=0.4)
#     optimizer = build_optimizer(model, cfg.optimizer)
    

#         # 데이터셋 transform 설정
#     train_transform = A.Compose([
#         A.Resize(512, 512)
#     ])
    
#     valid_transform = A.Compose([
#         A.Resize(512, 512)
#     ])
    
#     # 데이터셋 생성
#     train_dataset = XRayDataset(args.image_root, args.label_root, 
#                                is_train=True, transforms=train_transform)
#     valid_dataset = XRayDataset(args.image_root, args.label_root, 
#                                is_train=False, transforms=valid_transform)
    
#     # 데이터 로더 설정
#     train_loader = DataLoader(
#         dataset=train_dataset, 
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=8,
#         drop_last=True,
#         pin_memory=True
#     )
    
#     valid_loader = DataLoader(
#         dataset=valid_dataset, 
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=8,
#         drop_last=False,
#         pin_memory=True
#     )


#     # Learning rate scheduler 설정
#     total_iters = args.num_epochs * len(train_loader)
#     scheduler = PolynomialLR(
#         optimizer,
#         total_iters=total_iters,
#         power=args.power,
#         warmup_epochs=5,
#         num_epochs=args.num_epochs,
#         use_warmup=True
#     )
    
#     # Training with Hook
#     if args.use_hook:
#         with FeatureExtractor(model_with_hook, target_layers=args.target_layers) as feature_extractor:
#             train(model_with_hook, train_loader, valid_loader, criterion, optimizer,
#                   args.num_epochs, args.val_every, args.saved_dir, args.model_name,
#                   wandb=wandb, accumulation_steps=args.accumulation_steps,
#                   scheduler=scheduler)
#     else:
#         train(model_with_hook, train_loader, valid_loader, criterion, optimizer,
#               args.num_epochs, args.val_every, args.saved_dir, args.model_name,
#               wandb=wandb, accumulation_steps=args.accumulation_steps,
#               scheduler=scheduler)
        
# if __name__ == '__main__':
#     main()

# import os
# import argparse
# import wandb
# from mmengine.config import Config
# from mmseg.models import build_segmentor
# from mmengine.runner import Runner

# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a segmentor')
#     parser.add_argument('--config', default='/data/ephemeral/home/deamin/mmsegmentation/configs/deeplabv3plus/deeplabv3plus-swin-tiny.py',
#                       help='train config file path')
#     parser.add_argument('--work-dir', default='./work_dirs',
#                       help='the dir to save logs and models')
#     parser.add_argument('--wandb-project', default='FCN_baseline_deamin',
#                       help='Wandb project name')
#     parser.add_argument('--wandb-entity', default='cv01-HandBone-seg',
#                       help='Wandb team/organization name')
#     return parser.parse_args()

# def main():
#     args = parse_args()
    

#     # config 로드 및 수정
#     cfg = Config.fromfile(args.config)
    
#     # 작업 디렉토리 설정
#     cfg.work_dir = args.work_dir
#     os.makedirs(args.work_dir, exist_ok=True)
#         # pretrained 설정 수정
#     cfg.model.pretrained = None  # segmentor의 pretrained 제거
    
#     # 모델 생성
#     model = build_segmentor(cfg.model)
    
#     # Runner 생성 및 학습 시작
#     runner = Runner.from_cfg(cfg)
#     runner.train()

# if __name__ == '__main__':
#     main()