import cv2
import numpy as np
from albumentations import Lambda
from albumentations.core.transforms_interface import ImageOnlyTransform

def unsharp_mask(image, alpha=1.8, beta=-0.5, gamma=5):
        blurred = cv2.GaussianBlur(image, (3,3), sigmaX=12)
        return cv2.addWeighted(image, alpha, blurred, beta, gamma)

class UnsharpMask(ImageOnlyTransform):
        def __init__(self, always_apply=False, p=1,alpha=1.8, beta=-0.5, gamma=5):
                super(UnsharpMask, self).__init__(always_apply, p)
                self.alpha = alpha
                self.beta = beta
                self.gamma = gamma
        
        def apply(self, img, **params):
                img = (img * 255).astype(np.uint8)  # Albumentations 입력 이미지는 float이므로 uint8로 변환
                
                # RGB 또는 Grayscale 처리
                if img.ndim == 3:  # 컬러 이미지
                        processed_image = unsharp_mask(img, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
                elif img.ndim == 2:  # 그레이스케일 이미지
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        processed_image = unsharp_mask(img, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
                else:
                        raise ValueError(f"Unexpected image dimensions: {img.ndim}")
                
                return processed_image / 255.0  # Albumentations는 float32 [0, 1] 형식을 기대