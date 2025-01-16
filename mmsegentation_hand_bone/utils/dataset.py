from numba import njit
import cv2
from pathlib import Path
import numpy as np

@njit
def create_label_mask(mask_shape, annotations):
    """
    마스크 생성을 위한 최적화된 함수
    """
    label = np.zeros(mask_shape, dtype=np.float32)
    for points in annotations:
        points = points.astype(np.int32)
        cv2.fillPoly(label, [points], 1)
    return label

class XRayDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
        # 데이터 캐싱
        self.image_paths = []
        self.mask_cache = {}  # 마스크 캐시 저장소
        
        # 미리 모든 이미지 경로 수집
        self._load_image_paths()
        
        # 멀티프로세싱을 이용한 마스크 프리로딩
        if mode == 'train':
            self._preload_masks()
    
    def _load_image_paths(self):
        """이미지 경로를 미리 수집"""
        data_dir = Path(self.data_dir)
        self.image_paths = sorted(list(data_dir.glob('*/*.jpg')))
    
    def _preload_masks(self):
        """멀티프로세싱을 이용한 마스크 프리로딩"""
        from multiprocessing import Pool
        from functools import partial
        
        def load_single_mask(image_path):
            # 마스크 로딩 로직
            json_path = str(image_path).replace('.jpg', '.json')
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            
            # 마스크 생성
            mask = create_label_mask(mask_shape, annotations)
            return str(image_path), mask
        
        # 멀티프로세싱으로 마스크 로드
        with Pool() as pool:
            results = pool.map(load_single_mask, self.image_paths)
        
        # 결과를 캐시에 저장
        for path, mask in results:
            self.mask_cache[path] = mask
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # 이미지 로드 최적화
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 캐시된 마스크 사용
        mask = self.mask_cache.get(str(image_path))
        if mask is None:
            # 캐시에 없는 경우에만 로드
            json_path = str(image_path).replace('.jpg', '.json')
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            mask = create_label_mask(image.shape[:2], annotations)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask 