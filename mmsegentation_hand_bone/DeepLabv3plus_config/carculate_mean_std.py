import os
import numpy as np
from PIL import Image

def calculate_dataset_statistics(image_folder):
    pixel_sum = 0
    pixel_squared_sum = 0
    num_pixels = 0
    
    # 재귀적으로 모든 하위 폴더 탐색
    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.dcm')):
                image_path = os.path.join(root, filename)
                # Gray-scale로 이미지 로드
                img = np.array(Image.open(image_path).convert('L')).astype(np.float32)
                
                pixel_sum += img.sum()
                pixel_squared_sum += (img ** 2).sum()
                num_pixels += img.size
    
    # 평균 계산
    mean = pixel_sum / num_pixels
    # 표준편차 계산
    std = np.sqrt((pixel_squared_sum / num_pixels) - (mean ** 2))
    
    return mean, std

# 사용 예시
image_folder = '../data/train/DCM'
mean, std = calculate_dataset_statistics(image_folder)
print(f'Dataset mean: {mean:.2f}')
print(f'Dataset std: {std:.2f}')