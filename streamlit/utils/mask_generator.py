import numpy as np
import cv2

# commit
import csv
import sys
csv.field_size_limit(sys.maxsize)  # 필드 크기 제한 증가

# 클래스를 모듈 레벨에서 사용 가능하도록 export
__all__ = ['MaskGenerator', 'PointCloudGenerator']

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
        
    @staticmethod
    def decode_rle_to_mask(rle, height, width):
        """
        RLE 인코딩된 마스크를 디코딩합니다.
        Args:
            rle: RLE 인코딩된 문자열
            height: 출력 이미지의 높이
            width: 출력 이미지의 너비
        Returns:
            디코딩된 마스크 (height x width) 또는 빈 마스크 (오류 발생 시)
        """
        try:
            if isinstance(rle, float):  # NaN 값 체크
                print(f"Warning: RLE is float value (possibly NaN)")
                return np.zeros((height, width), dtype=np.uint8)
                
            s = str(rle).strip().split()
            starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
            starts -= 1  # 1-based index를 0-based index로 변환
            ends = starts + lengths
            img = np.zeros(height * width, dtype=np.uint8)
            
            for lo, hi in zip(starts, ends):
                img[lo:hi] = 1
            
            return img.reshape(height, width)
            
        except Exception as e:
            print(f"Warning: Failed to decode RLE: {str(e)}")
            return np.zeros((height, width), dtype=np.uint8)
    
    @staticmethod
    def load_and_process_masks(data_loader, csv_path, image_name, image_shape, class_idx=None):  # self 파라미터 제거
        # CSV 파일 로드
        df = data_loader.load_inference_csv(csv_path)
        #print(df)
        # 선택된 이미지에 대한 마스크 정보만 필터링
        image_masks = df[df['image_name'] == image_name]
        print(image_name)
        # 전체 마스크 초기화
        combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # 각 클래스별 마스크 생성 및 결합
        for _, row in image_masks.iterrows():
            class_name = row['class']
            rle = row['rle']
            class_idx = class_idx if class_idx is not None else CLASSES.index(class_name) + 1
            
            # RLE 디코딩
            class_mask = MaskGenerator.decode_rle_to_mask(rle, image_shape[0], image_shape[1])
            combined_mask[class_mask == 1] = class_idx
        
        return combined_mask
    
    @staticmethod
    def load_and_process_masks_by_class(data_loader, csv_path, image_name, image_shape, target_class):
        """특정 클래스에 대한 마스크만 생성하는 함수"""
        # CSV 파일 로드
        df = data_loader.load_inference_csv(csv_path)
        
        # 선택된 이미지와 클래스에 대한 마스크 정보만 필터링
        image_masks = df[(df['image_name'] == image_name) & (df['class'] == target_class)]
        
        # 마스크 초기화
        class_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # 해당 클래스의 마스크만 처리
        for _, row in image_masks.iterrows():
            rle = row['rle']
            # RLE 디코딩
            decoded_mask = MaskGenerator.decode_rle_to_mask(rle, image_shape[0], image_shape[1])
            class_mask[decoded_mask == 1] = 1
        
        return class_mask

class PointCloudGenerator:
    @staticmethod
    def create_point_cloud_from_json(json_data, image_shape, alpha=0.1):
        # 포인트 클라우드를 저장할 이미지 생성
        point_cloud = np.zeros((*image_shape[:2], 3), dtype=np.float32)
        
        # annotation의 폴리곤 채우기
        for annotation in json_data['annotations']:
            points = np.array(annotation['points'])
            class_name = annotation['label']
            class_idx = CLASSES.index(class_name)
            
            # 임시 마스크 생성
            temp_mask = np.zeros(image_shape[:2], dtype=np.uint8)
            
            # 폴리곤 채우기
            cv2.fillPoly(temp_mask, [points.astype(np.int32)], 1)
            
            # 색상 적용
            for i in range(3):
                point_cloud[:, :, i][temp_mask == 1] = PALETTE[class_idx][i] * alpha
        
        return point_cloud.astype(np.uint8)

    @staticmethod
    def overlay_multiple_point_clouds(point_clouds):
        """여러 이미지의 폴리곤을 하나로 합치기"""
        if not point_clouds:
            return None
            
        result = np.zeros_like(point_clouds[0], dtype=np.float32)
        
        # 모든 폴리곤 합치기
        for cloud in point_clouds:
            mask = (cloud > 0).any(axis=2)
            result[mask] += cloud[mask]
            
        # 값 범위 조정 및 타입 변환
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    @staticmethod
    def mask_to_rgb(mask):
        # RGB 마스크 생성
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        # 각 클래스별로 해당하는 색상 적용
        for i, color in enumerate(PALETTE, start=1):  # 1부터 시작 (0은 배경)
            rgb_mask[mask == i] = color
            
        return rgb_mask
    
    @staticmethod
    def create_class_point_cloud(json_data_list, image_shape, alpha=0.1):
        """여러 이미지의 클래스별 포인트 클라우드 생성"""
        # 클래스별 포인트 클라우드를 저장할 딕셔너리
        class_clouds = {class_name: np.zeros((*image_shape[:2], 3), dtype=np.float32) 
                       for class_name in CLASSES}
        
        # 모든 JSON 데이터에 대해 처리
        for json_data in json_data_list:
            for annotation in json_data['annotations']:
                points = np.array(annotation['points'])
                class_name = annotation['label']
                class_idx = CLASSES.index(class_name)
                
                # 임시 마스크 생성
                temp_mask = np.zeros(image_shape[:2], dtype=np.uint8)
                cv2.fillPoly(temp_mask, [points.astype(np.int32)], 1)
                
                # 해당 클래스의 포인트 클라우드에 추가
                for i in range(3):
                    class_clouds[class_name][:, :, i][temp_mask == 1] += PALETTE[class_idx][i] * alpha
        
        # 각 클래스별 포인트 클라우드 정규화
        for class_name in CLASSES:
            class_clouds[class_name] = np.clip(class_clouds[class_name], 0, 255).astype(np.uint8)
            
        return class_clouds