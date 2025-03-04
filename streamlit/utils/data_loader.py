import os
from PIL import Image
import json

# commit 
import pandas as pd

class DataLoader:
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.images_dir = os.path.join(data_dir, mode, "DCM")
        # test 모드에서는 json 디렉토리를 설정하지 않음
        self.json_dir = os.path.join(data_dir, mode, "outputs_json") if mode == 'train' else None
        
    def get_image_list(self):
        image_files = []
        
        for root, dirs, files in os.walk(self.images_dir):
            
            for f in files:
                if f.endswith('.png'):
                    rel_path = os.path.relpath(root, self.images_dir)
                    full_path = os.path.join(rel_path, f)
                    image_files.append(full_path)
                    # print(f"Added image: {full_path}")  # 추가된 이미지 경로
        
        return sorted(image_files)
    
    def load_image(self, image_path):
        full_path = os.path.join(self.images_dir, image_path)
        return Image.open(full_path)

    def get_json_path(self, image_path):
        if self.mode != 'train':
            raise ValueError("JSON files are only available in train mode")
        image_dir, image_name = os.path.split(image_path)
        json_name = image_name.replace('.jpg', '.json').replace('.png', '.json')
        return os.path.join(self.json_dir, image_dir, json_name)

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def get_image_pairs(self, image_files):
        """폴더별로 이미지 쌍을 찾아서 반환하는 함수"""
        pairs = {}
        for file in image_files:
            # 파일의 디렉토리 경로와 파일명을 분리
            dir_path = os.path.dirname(file)
            
            # 같은 폴더에 있는 파일들을 쌍으로 묶음
            if dir_path not in pairs:
                pairs[dir_path] = {'L': None, 'R': None}
            
            # 파일명에 따라 L/R 구분
            # 예시: 첫 번째 파일을 L, 두 번째 파일을 R로 지정
            if pairs[dir_path]['L'] is None:
                pairs[dir_path]['L'] = file
            else:
                pairs[dir_path]['R'] = file
                
        # L과 R 이미지가 모두 있는 쌍만 반환
        return {k: v for k, v in pairs.items() if v['L'] is not None and v['R'] is not None}
    
    def load_inference_csv(self, csv_path):
        if self.mode != 'test':
            raise ValueError("CSV files are only available in test mode")
        return pd.read_csv(csv_path)