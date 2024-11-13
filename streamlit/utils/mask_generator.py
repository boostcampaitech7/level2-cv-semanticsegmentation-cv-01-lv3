import numpy as np
import cv2

# 색상 리스트
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# 상수 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

class MaskGenerator:
    @staticmethod
    def create_mask_from_json(json_data, image_shape):
        # 흑백 마스크 생성 (클래스 인덱스용)
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for annotation in json_data['annotations']:
            points = np.array(annotation['points'])
            class_name = annotation['label']  # annotation에서 클래스 이름 가져오기
            class_idx = CLASSES.index(class_name) + 1  # 클래스 인덱스 (0은 배경)
            cv2.fillPoly(mask, [points.astype(np.int32)], class_idx)
        
        return mask

    @staticmethod
    def mask_to_rgb(mask):
        # RGB 마스크 생성
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        # 각 클래스별로 해당하는 색상 적용
        for i, color in enumerate(PALETTE, start=1):  # 1부터 시작 (0은 배경)
            rgb_mask[mask == i] = color
            
        return rgb_mask

# 클래스를 모듈 레벨에서 사용 가능하도록 export
__all__ = ['MaskGenerator']