import json
import os
import numpy as np
from PIL import Image, ImageDraw
import shutil
from tqdm import tqdm

# 상수 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def convert_dataset(dcm_root, json_root, output_root):
    """데이터셋을 mmsegmentation 형식으로 변환"""
    # 출력 디렉토리 생성
    os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'annotations'), exist_ok=True)
    
    # ID 폴더들을 순회
    for id_folder in tqdm(os.listdir(dcm_root)):
        dcm_id_path = os.path.join(dcm_root, id_folder)
        json_id_path = os.path.join(json_root, id_folder)
        
        if not os.path.isdir(dcm_id_path):
            continue
            
        # 각 이미지에 대해 처리
        for img_file in os.listdir(dcm_id_path):
            name = os.path.splitext(img_file)[0]
            
            # 이미지 복사
            src_img = os.path.join(dcm_id_path, img_file)
            dst_img = os.path.join(output_root, 'images', f"{id_folder}_{img_file}")
            
            # 원본 이미지 크기 얻기
            img_size = Image.open(src_img).size
            
            # 이미지 복사
            shutil.copy2(src_img, dst_img)
            
            # JSON을 멀티채널 마스크로 변환
            json_file = os.path.join(json_id_path, f"{name}.json")
            if os.path.exists(json_file):
                masks = convert_json_to_multimask(json_file, img_size)
                mask_path = os.path.join(output_root, 'annotations', f"{name}.npz")
                np.savez_compressed(mask_path, masks=masks)

def convert_json_to_multimask(json_path, img_size):
    """JSON 파일을 멀티채널 마스크로 변환"""
    # 각 클래스별로 별도의 채널 생성 (클래스 수만큼의 채널)
    masks = np.zeros((len(CLASSES), img_size[1], img_size[0]), dtype=np.uint8)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for ann in data['annotations']:
        if ann['type'] == 'poly_seg':
            label = ann['label']
            if label in CLASS2IND:
                class_idx = CLASS2IND[label]
                points = np.array(ann['points'], dtype=np.int32)
                
                # 해당 클래스 채널에 마스크 그리기
                temp_mask = Image.new('L', (img_size[0], img_size[1]), 0)
                draw = ImageDraw.Draw(temp_mask)
                draw.polygon([tuple(point) for point in points], fill=1)
                masks[class_idx] = np.array(temp_mask)
    
    return masks

if __name__ == "__main__":
    # 경로 설정
    dcm_root = "../data/train/DCM"
    json_root = "../data/train/outputs_json"
    output_root = "../data/mmseg_format_npz"

    # 변환 실행
    convert_dataset(dcm_root, json_root, output_root)
        
# train/
#   ├── DCM/
#   │   ├── ID001/
#   │       ├── image1.jpg
#   │       ├── image2.jpg
#   │   ├── ID002/
#   │       ├── image1.jpg
#   │       ├── image2.jpg
#   │
#   └── outputs_json/
#       ├── ID001/
#       │   ├── image1.json  # 변환된 마스크
#       │   ├── image2.json
#       │
#       ├── ID002/
#       │   ├── image1.json
#       │   ├── image2.json
            