import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class Augmentations:
    def __init__(self):
        self.transforms = {
            "기본 변환": self.get_basic_transforms(),
            "노이즈 추가": self.get_noise_transforms(),
            "기하학적 변환": self.get_geometric_transforms(),
            "사용자 정의 변환": self.get_custom_transforms(),
            "히스토그램 평활화": self.get_histogram_transforms(),  # 새로운 변환 추가
        }
    
    def get_basic_transforms(self):
        return A.Compose([
            A.RandomBrightnessContrast(p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
    
    def get_noise_transforms(self):
        return A.Compose([
            A.GaussNoise(p=0.8),
            A.ISONoise(p=0.8),
            A.MultiplicativeNoise(p=0.8),
        ])
    
    def get_geometric_transforms(self):
        return A.Compose([
            A.ShiftScaleRotate(p=0.8),
            A.ElasticTransform(p=0.8),
            A.GridDistortion(p=0.8),
        ])
    
    def get_custom_transforms(self):
        return A.OneOf([
            A.RandomBrightnessContrast(p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], p=0.1)
    
    def apply_transform(self, image, transform_name):
        if transform_name not in self.transforms:
            return image
        
        transformed = self.transforms[transform_name](image=image)
        return transformed["image"]
    
    def get_histogram_transforms(self):
        return A.Compose([
            A.CLAHE(clip_limit=3.0, tile_grid_size=(3, 3), p=1.0),  # Contrast Limited Adaptive Histogram Equalization
            #A.Equalize(p=1.0),  # 일반적인 히스토그램 평활화
        ])
    
    