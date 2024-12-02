from mmengine.runner import Runner
from mmengine.config import Config
import wandb
from mmengine.hooks import Hook
from mmseg.registry import HOOKS
import numpy as np
from utils.method import dice_coef, calculate_confusion_matrix
import plotly.graph_objects as go

import matplotlib.pyplot as plt

import argparse

# ArgumentParser 추가
# parser = argparse.ArgumentParser(description='Train a segmentor')
# parser.add_argument('--resume', type=str, help='체크포인트 경로', default='/data/ephemeral/home/deamin/backup/mmsegmentation/work_dirs/segformer_mit-b5_xray_MultiStepLR/iter_2500.pth')
# args = parser.parse_args()
# 상수 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
@HOOKS.register_module()
class CustomWandbHook(Hook):
    def __init__(self, by_epoch=False, interval=10):  # 로깅 간격 파라미터 추가
        self.by_epoch = by_epoch
        self.interval = interval  # 로깅 간격 설정
        self.dice_history = {class_name: [] for class_name in CLASSES}
        self.epoch_history = []
        
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        # 학습 중 손실값 로깅
        if batch_idx % self.interval == 0:
            current_lr = runner.optim_wrapper.get_lr()
            # 리스트인 경우 첫 번째 값만 사용하도록 수정
            if isinstance(current_lr, (list, tuple)):
                current_lr = float(current_lr[0])  # float로 명시적 변환
            
            wandb.log({
                "train/loss": outputs['loss'].item(),
                "train/step": runner.iter,
                "train/epoch": runner.epoch,
                "train/learning_rate": runner.optim_wrapper.get_lr(),  # 키 이름도 단순화
            })
    def after_val_epoch(self, runner, metrics=None):
        if not metrics:
            return
            
        # 기본 metrics 로깅
        wandb_metrics = {
            'valid/mean_dice': metrics['mDice'],
            'valid/epoch': runner.epoch
        }
        
        # 클래스별 Dice score만 로깅
        for class_name in CLASSES:
            wandb_metrics[f'valid/dice_{class_name}'] = metrics[f'dice_{class_name}']
        
        # Dice Scores 시각화
        dice_fig = go.Figure()
        for i, class_name in enumerate(CLASSES):
            dice_fig.add_trace(go.Bar(
                name=class_name,
                x=[class_name],
                y=[metrics[f'dice_{class_name}']]
            ))
        dice_fig.update_layout(
            title='Class-wise Dice Scores',
            yaxis_title='Dice Score',
            showlegend=True,
            height=800,  # 그래프 높이 증가
            xaxis_tickangle=-45  # x축 라벨 회전
        )
        
        # 시각화 결과 로깅
        wandb_metrics.update({
            'valid/dice_scores': dice_fig,
        })
        
        # wandb에 로깅
        wandb.log(wandb_metrics)

# Wandb 초기화
wandb.init(
    project="FCN_baseline_deamin",
    entity="cv01-HandBone-seg",
    name="mmseg_segformer_mit-b3-MultiStepLR_1536_20000_CLAHE_loss_0.5_lr_5e-4_EMA"
)

# 설정 파일 로드
cfg = Config.fromfile('configs/segformer/segformer_mit-b3_xray.py')
cfg.work_dir = './work_dirs/segformer_mit-b3_xray_MultiStepLR_1536_20000_CLAHE_loss_0.5_lr_5e-4_EMA'

# Custom Hook 등록 (interval 값 지정)
cfg.custom_hooks = [
    dict(type='CustomWandbHook', priority='NORMAL', interval=10),
    dict(type='EmptyCacheHook', priority='NORMAL', after_val=True, after_train_epoch=False)
]

# # resume 옵션 추가
# if args.resume:
#     cfg.resume = True
#     cfg.load_from = args.resume  # 체크포인트에서 가중치 로드
#     print(f'체크포인트를 불러옵니다: {args.resume}')

# 학습 실행
runner = Runner.from_cfg(cfg)
runner.train()

# Wandb 종료
wandb.finish()