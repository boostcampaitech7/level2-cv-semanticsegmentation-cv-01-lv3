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
            "엣지 강화": self.get_edge_enhancement_transforms(),  # 새로운 변환 추가
            "그래디언트 강화": self.get_bone_edge_transforms(),  # 새로운 변환 추가
            "경계 검출": self.get_boundary_detection(),  # 새로운 변환 추가
        }
        self.boundary_detector = BoundaryDetection(theta0=3, theta=5)
    
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
    
    def get_edge_enhancement_transforms(self):
        return A.Compose([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, p=1.0),
        ])

    def get_bone_edge_transforms(self):
        def enhance_bone_edges(image, methods=['laplacian', 'clahe'], **kwargs):
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            enhanced = gray.astype(np.float64)
            
            if 'canny' in methods:
                canny = cv2.Canny(gray, threshold1=30, threshold2=150)
                enhanced = cv2.addWeighted(enhanced.astype(np.float32), 1.0, canny.astype(np.float32), 0.3, 0)
                
            if 'scharr' in methods:
                scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
                scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
                scharr = np.sqrt(scharrx**2 + scharry**2)
                enhanced = cv2.addWeighted(enhanced.astype(np.float32), 1.0, 
                                        cv2.convertScaleAbs(scharr).astype(np.float32), 0.3, 0)
                
            if 'log' in methods:
                blur = cv2.GaussianBlur(gray, (3,3), 0)
                log = cv2.Laplacian(blur, cv2.CV_64F)
                enhanced = cv2.addWeighted(enhanced.astype(np.float32), 1.0, 
                                        cv2.convertScaleAbs(log).astype(np.float32), 0.2, 0)
                
            if 'dog' in methods:
                gaussian1 = cv2.GaussianBlur(gray, (3,3), 1.0)
                gaussian2 = cv2.GaussianBlur(gray, (3,3), 2.0)
                dog = gaussian1 - gaussian2
                enhanced = cv2.addWeighted(enhanced.astype(np.float32), 1.0, 
                                        cv2.convertScaleAbs(dog).astype(np.float32), 0.2, 0)
                
            if 'sobel' in methods:
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx**2 + sobely**2)
                enhanced = cv2.addWeighted(enhanced.astype(np.float32), 1.0, 
                                        cv2.convertScaleAbs(sobel).astype(np.float32), 0.3, 0)
                
            if 'laplacian' in methods:
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                enhanced = cv2.addWeighted(enhanced.astype(np.float32), 1.0, 
                                        cv2.convertScaleAbs(laplacian).astype(np.float32), 0.2, 0)
                
            if 'clahe' in methods:
                # CLAHE 적용
                enhanced_uint8 = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(enhanced_uint8)
            
            # 최종 결과를 0-255 범위로 변환
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            enhanced = enhanced.astype(np.uint8)

            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                enhanced = enhanced
                
            return enhanced

        return A.Lambda(
            image=enhance_bone_edges,
            p=1.0
        )
    
    def get_boundary_detection(self):
        return lambda x: self.boundary_detector.get_boundary(x)
    
    def apply_transform(self, image, transform_name):
        if transform_name not in self.transforms:
            return image
        
        if transform_name == "경계 검출":
            # 경계 검출은 특별한 처리가 필요
            boundary = self.transforms[transform_name](image)
            # 시각화를 위해 [0, 255] 범위로 정규화
            boundary = (boundary * 255).astype(np.uint8)
            # 단일 채널인 경우 3채널로 변환
            if len(boundary.shape) == 2:
                boundary = cv2.cvtColor(boundary, cv2.COLOR_GRAY2RGB)
            return boundary
        else:
            transformed = self.transforms[transform_name](image=image)
            return transformed["image"]
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryDetection:
    def __init__(self, theta0=2, theta=2):
        self.theta0 = theta0
        self.theta = theta
        
    def get_boundary(self, image):
        # numpy 배열을 torch tensor로 변환
        if isinstance(image, np.ndarray):
            x = torch.from_numpy(image).float()
            if len(x.shape) == 2:
                x = x.unsqueeze(0).unsqueeze(0)
            elif len(x.shape) == 3:
                x = x.permute(2, 0, 1).unsqueeze(0)
        
        # 다중 스케일 Sobel 필터 정의
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # 라플라시안 필터 추가
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        
        # 필터 준비
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        laplacian = laplacian.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        
        # 다중 스케일 경계 맵 계산
        gx = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
        gy = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
        gl = F.conv2d(x, laplacian, padding=1, groups=x.size(1))
        
        # 경계 강도 계산 (Sobel + Laplacian)
        edge_sobel = torch.sqrt(gx**2 + gy**2)
        edge_laplacian = torch.abs(gl)
        
        # 두 특징 결합
        edge = edge_sobel * 0.3 + edge_laplacian * 0.7
        
        # 적응형 임계값 적용
        mean_edge = torch.mean(edge)
        std_edge = torch.std(edge)
        lower_threshold = mean_edge - 0.5 * std_edge
        upper_threshold = mean_edge + 2 * std_edge
        
        # 임계값 기반 정규화
        edge = torch.clamp((edge - lower_threshold) / (upper_threshold - lower_threshold), 0, 1)
        
        # torch tensor를 numpy 배열로 변환
        edge = edge.squeeze().numpy()
        if len(edge.shape) == 3:
            edge = np.transpose(edge, (1, 2, 0))
        
        return edge