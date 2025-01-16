_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/hand_xray.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# Wandb import(Feature:#3 Wandb logging, deamin, 2024.11.12)
#import wandb
# 상수 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
# Swin Transformer 설정

from mmseg.utils import register_all_modules
register_all_modules()

# Swin Transformer 설정
model = dict(
    backbone=dict(
        _delete_=True,  # 기존 ResNet backbone 설정 삭제
        type='SwinTransformer',
        embed_dims=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'),
    decode_head=dict(
        in_channels=768,  # Swin-T의 마지막 stage 출력 채널 수
        c1_in_channels=96  # Swin-T의 첫 번째 stage 출력 채널 수
    ),
    auxiliary_head=dict(
        in_channels=384  # Swin-T의 세 번째 stage 출력 채널 수
    ))

# 옵티마이저 설정
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,  # train.py와 동일한 학습률로 수정
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# 학습 스케줄러 설정 수정 (PolynomialLR과 유사하게)
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001,
        by_epoch=False, 
        begin=0,
        end=1000),
    dict(
        type='PolyLR',
        power=0.9,  # train.py의 power 값과 동일하게 설정
        begin=1000,
        end=40000,
        eta_min=0.0,
        by_epoch=False)
]

# 데이터 설정
dataset_type = 'HandXRayDataset'
data_root = '/data/ephemeral/home/deamin/backup/level2-cv-semanticsegmentation-cv-01-lv3/data/'

# 데이터 파이프라인 수정 (이미지 크기 512x512로 변경)
# 데이터 파이프라인 수정
# 데이터 파이프라인 수정
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(512, 512)),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512))
]

# 데이터로더 설정 수정
train_dataloader = dict(
    batch_size=4,  # train.py와 동일한 배치 사이즈
    num_workers=8,  # train.py와 동일한 worker 수
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        data_prefix=dict(img_path='train/DCM', seg_map_path='train/outputs_json'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        data_prefix=dict(img_path='train/DCM', seg_map_path='train/outputs_json'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Wandb 로깅 설정
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='FCN_baseline_deamin',  # wandb project 이름
                 entity='cv01-HandBone-seg',    # wandb entity 이름
                 name='deeplabv3plus-swin-tiny' # 실험 이름
             ),
             # 로깅할 지표들 정의
             log_metrics_by_epoch=True,
             log_checkpoint=True,
             log_model_weights=True)
    ])

# Evaluator 설정
val_evaluator = [
    dict(type='IoUMetric',
         iou_metrics=['mIoU'],
         prefix='val',
         per_class_metrics=True),  # 클래스별 IoU 계산
    dict(type='DiceMetric',
         prefix='val',
         per_class_metrics=True)   # 클래스별 Dice 계산
]
test_evaluator = val_evaluator

# Visualizer 설정
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend')
    ],
    name='visualizer',
    save_dir='visual_results'
)

# default_hooks 설정
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1000, max_keep_ckpts=3, save_best='mDice'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=100)
)

# 평가 설정
evaluation = dict(
    interval=1,  # 평가 주기
    metric=['mIoU', 'mDice'],  # 평가 메트릭
    save_best='mDice',  # 최고 성능 모델 저장 기준
    pre_eval=True  # 사전 평가 수행
)

# 체크포인트 설정
checkpoint_config = dict(
    interval=1000,  # 체크포인트 저장 주기
    max_keep_ckpts=3,  # 최대 저장 체크포인트 수
    save_best='mDice',  # 최고 성능 모델 저장 기준
)