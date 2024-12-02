import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

class FeatureExtractor():
    def __init__(self, model, target_layers=None, model_type=None):
        """
        Args:
            model: 특징을 추출할 모델
            target_layers: 특징을 추출할 레이어 이름 리스트. None이면 모델 타입에 따라 자동 설정
            model_type: 모델 타입 ('fcn', 'unet', 'deeplabv3' 등). None이면 자동 감지 시도
        """
        self.model = model
        self.model_type = self._detect_model_type(model) if model_type is None else model_type
        
        # # 모델 구조 출력
        # for name, module in model.named_modules():
        #     print(f"Layer: {name}")
        
        self.target_layers = [
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
        ]
            
        self.features = {}  # 빈 딕셔너리로 초기화
        self.hooks = []
        self._register_hooks()  # 단일 hook 등록 방식 사용

    def _register_hooks(self):
        """훅 등록을 위한 단일 메서드"""
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(
                    lambda m, i, o, name=name: self._hook_fn(name, o)
                )
                self.hooks.append(hook)

    def _hook_fn(self, name, output):
        """통일된 hook 함수"""
        # classifier 출력 처리
        if 'classifier' in name:
            if isinstance(output, dict) and 'out' in output:
                output = output['out']
            elif isinstance(output, (tuple, list)):
                output = output[0]
        
        # 배치 차원이 없는 경우 추가
        if len(output.shape) == 3:
            output = output.unsqueeze(0)
        
        # 배치의 첫 번째 이미지에 대한 feature map만 저장
        self.features[name] = output[0:1].detach().cpu()

    def _detect_model_type(self, model):
        """모델 타입 자동 감지"""
        model_name = model.__class__.__name__.lower()
        if 'fcn' in model_name:
            return 'fcn'
        elif 'unet' in model_name:
            return 'unet'
        elif 'deeplabv3' in model_name:
            return 'deeplabv3'
        else:
            return 'custom'

    def _get_default_target_layers(self):
        """모델 타입에 따른 기본 타겟 레이어 설정"""
        default_layers = {
            'fcn': [
                'backbone.layer1',
                'backbone.layer2',
                'backbone.layer3',
                'backbone.layer4',
                'aux_classifier.4',
                'classifier.4'
            ],
            'unet': [
                'encoder1',
                'encoder2',
                'encoder3',
                'encoder4',
                'bottleneck'
            ],
            'deeplabv3': [
                'backbone.layer1',
                'backbone.layer2',
                'backbone.layer3',
                'backbone.layer4',
                'classifier.4'
            ],
            'custom': []  # 사용자 정의 필요
        }
        return default_layers.get(self.model_type, [])

    def clear_features(self):
        """특징 맵 초기화"""
        for layer in self.target_layers:
            self.features[layer] = []

    def remove_hooks(self):
        """등록된 훅 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_feature_maps(self, layer_name=None):
        """특정 레이어의 특징 맵 반환"""
        if layer_name is None:
            return self.features
        return self.features.get(layer_name, None)

    def visualize_features(self, wandb_logger=None, save_dir='./feature_maps', max_features=4):
        """Feature map 시각화 함수"""
        os.makedirs(save_dir, exist_ok=True)
        
        for layer_name, features in self.features.items():
            # features가 비어있는지 확인
            if features is None or features.numel() == 0:
                print(f"Skipping empty feature map for layer: {layer_name}")
                continue
                
            # 첫 번째 이미지의 feature map만 선택
            feature_map = features[0]  # [C, H, W]
            
            # 채널 수 확인
            num_channels = feature_map.size(0)
            if num_channels == 0:
                print(f"Skipping layer {layer_name} with 0 channels")
                continue
                
            # classifier나 aux_classifier의 경우 모든 채널 시각화
            if 'segmentation_head' in layer_name:
                num_features = num_channels
                # subplot 크기 계산 (가로 5개씩 표시)
                num_rows = (num_features + 4) // 5
                num_cols = min(5, num_features)
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
                axes = axes.flatten() if num_rows > 1 or num_cols > 1 else [axes]
            else:
                # 다른 레이어는 기존대로 max_features 만큼만 표시
                num_features = min(num_channels, max_features)
                fig, axes = plt.subplots(1, num_features, figsize=(num_features * 4, 4))
                if num_features == 1:
                    axes = [axes]
                    
            for idx in range(num_features):
                channel_data = feature_map[idx].cpu().numpy()
                
                # 데이터 정규화
                if channel_data.max() != channel_data.min():
                    channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
                
                # 시각화
                axes[idx].imshow(channel_data, cmap='viridis')
                axes[idx].axis('off')
                axes[idx].set_title(f'Class {idx}' if 'classifier' in layer_name else f'Channel {idx}')
            
            # 사용하지 않는 subplot 제거
            if 'classifier' in layer_name:
                for idx in range(num_features, len(axes)):
                    fig.delaxes(axes[idx])
                    
            plt.suptitle(f'Feature Maps for {layer_name}')
            plt.tight_layout()
            
            # 저장 및 로깅
            save_path = os.path.join(save_dir, f'feature_map_{layer_name}.png')
            plt.savefig(save_path)
            if wandb_logger:
                wandb_logger.log({f"feature_maps/{layer_name}": wandb.Image(save_path)})
            plt.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
