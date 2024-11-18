import cv2
import numpy as np
import albumentations as A
import os
import torch
from utils.dataset import XRayDataset

def save_side_by_side_comparison(dataset, idx, output_dir="./output_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터셋에서 이미지 가져오기
    original_image, _ = dataset[idx]
    
    # 디버깅: 이미지 정보 출력
    print(f"Image type: {type(original_image)}")
    print(f"Image shape: {original_image.shape}, dtype: {original_image.dtype}")
    
    # 이미지 NumPy 배열로 변환
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.permute(1, 2, 0).numpy()  # CxHxW -> HxWxC로 변환
    
    # (C, H, W) -> (H, W, C)로 변환
    if original_image.ndim == 3 and original_image.shape[0] == 3:
        original_image = np.transpose(original_image, (1, 2, 0))
    
    # CLAHE 변환 정의 (다양한 clip_limit 값 적용)
    clip_limits = [2.0, 4.0, 6.0]
    clahe_images = []
    
    for clip_limit in clip_limits:
        clahe_transform = A.Compose([A.CLAHE(p=1.0)])
        
        # CLAHE 적용
        augmented = clahe_transform(image=original_image)
        clahe_image = augmented['image']
        
        # CLAHE 강도 조정
        clahe_image = np.uint8(clahe_image * 255)  # CLAHE 적용 전, 이미지를 uint8로 변환
        
        # CLAHE 처리 (clip_limit 값 적용)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))  # CLAHE 강도를 설정
        clahe_image = clahe.apply(clahe_image)  # CLAHE 결과 적용
        clahe_images.append(clahe_image)
    
    # 결과 이미지를 나란히 배치
    original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)
    
    # 각 CLAHE 결과를 3채널 컬러로 변환하여 시각화
    clahe_images_colored = [cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) for image in clahe_images]
    
    # 3개의 이미지를 나란히 배치
    combined_image = np.hstack([original_image] + clahe_images_colored)
    
    # 저장 경로 설정
    comparison_path = os.path.join(output_dir, f"clahe_comparison_image_{idx}.png")
    
    # 이미지 저장
    cv2.imwrite(comparison_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    print(f"Comparison image saved to: {comparison_path}")

# 테스트 실행
if __name__ == "__main__":
    # 데이터셋 로드
    dataset = XRayDataset(image_root="./data/train/DCM", label_root="./data/train/outputs_json", is_train=True)
    
    # 비교 이미지 저장
    save_side_by_side_comparison(dataset, idx=0)
