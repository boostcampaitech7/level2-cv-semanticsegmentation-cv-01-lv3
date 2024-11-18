import os
import shutil

# 원본 데이터가 있는 디렉토리 (ID001, ID002 등 폴더가 있는 디렉토리 경로)
source_dir = "./data/train"  # 각 ID 폴더가 위치한 경로
image_dest = "./image"       # 이미지 파일이 모일 경로
label_dest = "./label"       # JSON 파일이 모일 경로

image_source_dir = os.path.join(source_dir, 'DCM')
json_source_dir = os.path.join(source_dir, 'outputs_json')

# 이미지와 라벨 파일을 저장할 폴더를 생성합니다.
os.makedirs(image_dest, exist_ok=True)
os.makedirs(label_dest, exist_ok=True)

# 이미지 파일 복사
for id_folder in os.listdir(image_source_dir):
    id_folder_path = os.path.join(image_source_dir, id_folder)
    
    # 폴더인지 확인
    if os.path.isdir(id_folder_path):
        # ID 폴더 안의 파일들을 탐색
        for filename in os.listdir(id_folder_path):
            file_path = os.path.join(id_folder_path, filename)
            
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 확장자에 맞춰 수정 가능
                shutil.copy2(file_path, os.path.join(image_dest, f"{id_folder}_{filename}"))
                
# JSON 파일 복사
for id_folder in os.listdir(json_source_dir):
    id_folder_path = os.path.join(json_source_dir, id_folder)
    
    # 폴더인지 확인
    if os.path.isdir(id_folder_path):
        # ID 폴더 안의 파일들을 탐색
        for filename in os.listdir(id_folder_path):
            file_path = os.path.join(id_folder_path, filename)
            
            if filename.endswith('.json'):
                shutil.copy2(file_path, os.path.join(label_dest, f"{id_folder}_{filename}"))

print("파일 복사가 완료되었습니다.")
