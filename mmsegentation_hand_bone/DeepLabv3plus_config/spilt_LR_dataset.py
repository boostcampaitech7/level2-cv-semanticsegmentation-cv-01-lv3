import os
import shutil
from pathlib import Path

class DatasetPairSeparator:
    def __init__(self, root_dir, output_base_dir):
        """
        Args:
            root_dir (str): 데이터가 있는 루트 디렉토리 (DCM과 outputs_json의 상위 폴더)
            output_base_dir (str): 분리된 데이터를 저장할 기본 디렉토리
        """
        self.root_dir = Path(root_dir)
        self.dcm_dir = self.root_dir / 'DCM'
        self.json_dir = self.root_dir / 'outputs_json'
        
        self.output_base_dir = Path(output_base_dir)
        self.left_dir = self.output_base_dir / 'left'
        self.right_dir = self.output_base_dir / 'right'
        
        # 출력 디렉토리 생성
        for data_type in ['DCM', 'outputs_json']:
            (self.left_dir / data_type).mkdir(parents=True, exist_ok=True)
            (self.right_dir / data_type).mkdir(parents=True, exist_ok=True)

    def get_image_pairs(self):
        """폴더별로 이미지 쌍을 찾아서 반환"""
        pairs = {}
        for root, _, files in os.walk(self.dcm_dir):
            png_files = [f for f in files if f.endswith('.png')]
            if not png_files:
                continue
                
            dir_path = Path(root).relative_to(self.dcm_dir)
            pairs[dir_path] = {'L': None, 'R': None}
            
            # 파일명 정렬하여 첫 번째는 L, 두 번째는 R로 할당
            sorted_files = sorted(png_files)
            if len(sorted_files) >= 1:
                pairs[dir_path]['L'] = sorted_files[0]
            if len(sorted_files) >= 2:
                pairs[dir_path]['R'] = sorted_files[1]
                
        return {k: v for k, v in pairs.items() if v['L'] is not None and v['R'] is not None}

    def separate_and_save_pairs(self):
        """이미지와 JSON 파일을 left/right 폴더로 분리하여 저장"""
        pairs = self.get_image_pairs()
        
        for folder_path, pair in pairs.items():
            for data_type, src_dir in [('DCM', self.dcm_dir), ('outputs_json', self.json_dir)]:
                # 원본 파일 경로
                left_src = src_dir / folder_path / pair['L']
                right_src = src_dir / folder_path / pair['R']
                
                # JSON 파일의 경우 확장자를 .json으로 변경
                if data_type == 'outputs_json':
                    left_src = left_src.with_suffix('.json')
                    right_src = right_src.with_suffix('.json')
                
                # 저장할 파일 경로
                left_dst = self.left_dir / data_type / folder_path / left_src.name
                right_dst = self.right_dir / data_type / folder_path / right_src.name
                
                # 대상 디렉토리 생성
                left_dst.parent.mkdir(parents=True, exist_ok=True)
                right_dst.parent.mkdir(parents=True, exist_ok=True)
                
                # 파일이 존재하는 경우에만 복사
                if left_src.exists():
                    shutil.copy2(left_src, left_dst)
                    print(f"복사됨 ({data_type} L): {left_src.name} -> {left_dst}")
                
                if right_src.exists():
                    shutil.copy2(right_src, right_dst)
                    print(f"복사됨 ({data_type} R): {right_src.name} -> {right_dst}")

def reorganize_files():
    # 기본 경로 설정
    base_path = Path('../train/right_temp')
    dcm_path = base_path / 'DCM'
    json_path = base_path / 'outputs_json'
    
    # 새로운 디렉토리 생성
    new_image_path = Path('../train/images')
    new_json_path = Path('../train/outputs_json')
    new_image_path.mkdir(parents=True, exist_ok=True)
    new_json_path.mkdir(parents=True, exist_ok=True)
    
    # ID 폴더들을 순회
    for id_folder in dcm_path.glob('ID*'):
        # 각 ID 폴더 내의 이미지 파일들을 순회
        for img_file in id_folder.glob('*.png'):
            # 원본 파일 이름 유지
            new_img_name = img_file.name
            new_json_name = f"{img_file.stem}.json"
            
            # 해당 ID에 대한 json 파일 경로
            json_file = json_path / id_folder.name / new_json_name
            
            # 파일 이동
            if img_file.exists():
                shutil.copy2(img_file, new_image_path / new_img_name)
            
            if json_file.exists():
                shutil.copy2(json_file, new_json_path / new_json_name)


def main():
    # 사용 예시
    root_dir = "../data/train"  # DCM과 outputs_json 폴더가 있는 경로
    output_dir = "../data/split/train"
    
    # separator = DatasetPairSeparator(root_dir, output_dir)
    # separator.separate_and_save_pairs()

    reorganize_files()

if __name__ == "__main__":
    main()