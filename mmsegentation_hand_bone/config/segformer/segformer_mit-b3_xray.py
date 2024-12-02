_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/xray.py',
    '../_base_/default_runtime.py'
]

# data preprocessor 설정 수정
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[30.85, 30.85, 30.85],
    std=[42.00, 42.00, 42.00],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32  # size 대신 size_divisor 사용
)
# checkpoint URL 변경
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_512x512_160k_ade20k/segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth'

model = dict(
    type='EncoderDecoderWithoutArgmax',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),
        embed_dims=64,  # 수정: MIT-B3는 64로 시작
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3]),  # MIT-B3 구조
    decode_head=dict(
        type='SegformerHeadWithoutAccuracy',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=29,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=0.5,
                ),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                loss_weight=0.5,
                naive_dice=True)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')  # threshold 추가
)

# optimizer
optimizer = dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
optim_wrapper = dict(
    type='AmpOptimWrapper',  # OptimWrapper에서 AmpOptimWrapper로 변경
    optimizer=optimizer,
    clip_grad=None,
    loss_scale='dynamic',    # dynamic loss scaling 사용
    accumulative_counts=4, 
    # loss_scale_config=dict(  # loss scale 설정
    #     init_scale=2**16,
    #     scale_factor=2,
    #     scale_window=2000)
    )

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    gamma=0.5,
    begin=0,
    end=80000,  # 100K iterations으로 설정
    milestones=[40000, 60000],  # lr: 1e-3 -> 2e-4
    by_epoch=False)

# 변경사항 설명:
# type을 'OptimWrapper'에서 'AmpOptimWrapper'로 변경했습니다.
# Mixed Precision 학습을 위한 loss scaling 설정을 추가했습니다:
# loss_scale: 'dynamic'으로 설정하여 동적 loss scaling 사용
# loss_scale_config: loss scaling 관련 세부 설정
# init_scale: 초기 스케일 값 (2^16)
# scale_factor: 스케일 조정 비율
# scale_window: loss scale 업데이트 주기

# param_scheduler = [
#     dict(
#         type='ReduceLROnPlateau',
#         mode='min',
#         factor=0.2,
#         patience=10,
#         threshold=1e-4,
#         min_lr=1e-6,
#         by_epoch=True
#     )
# ]

# model_wrapper 설정 추가
model_wrapper = dict(
    type='ModelEMA',
    module=model,
    momentum=0.9999,
    interval=200
)

# training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=200)  # max_iters 100K로 설정
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadXRayAnnotations'),
    dict(type='Resize', scale=(1536, 1536)),
    dict(
    type='CLAHE',
    clip_limit=2.0,
    tile_grid_size=(5, 5)),  # CLAHE 추가
    dict(type='TransposeAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadXRayAnnotations'),  # 이 줄 추가
    dict(type='Resize', scale=(1536, 1536)),
    dict(
        type='CLAHE',
        clip_limit=2.0,
        tile_grid_size=(5, 5)),  # CLAHE 추가
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='XRayDataset',
        is_train=True,
        data_root='path/to/your/data',
        data_prefix=dict(img_path='train/DCM', seg_map_path='train/outputs_json'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=3,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='XRayDataset',
        is_train=False,
        data_root='path/to/your/data',
        data_prefix=dict(img_path='train/DCM', seg_map_path='train/outputs_json'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# evaluation settings
val_evaluator = dict(type='DiceMetric')
test_evaluator = val_evaluator 

# checkpoint 설정
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,                 # 체크포인트 저장 간격
        max_keep_ckpts=3,             # 최대 보관할 체크포인트 수
        save_best=['mDice'],          # best model 저장 기준
        rule='greater'),              # 더 큰 값이 더 좋음
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))