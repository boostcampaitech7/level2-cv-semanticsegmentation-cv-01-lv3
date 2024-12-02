import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# 파일 상단에 상수 정의 추가
IMAGE_HEIGHT = 2048
IMAGE_WIDTH = 2048

def csv_ensemble(csv_paths, save_dir, class_csv_mapping, CLASS2IND):  # threshold 파라미터 추가
    def decode_rle_to_mask(rle, height, width):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(height * width, dtype=np.uint8)
        
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        
        return img.reshape(height, width)

    def encode_mask_to_rle(mask):
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    # csv의 기본 column(column이지만 사실 row입니다.. default 8352)
    csv_column = 8352

    csv_data = []
    for path in csv_paths:
        data = pd.read_csv(path)
        csv_data.append(data)

    file_num = len(csv_data)
    filename_and_class = []
    rles = []

    print(f"앙상블할 모델 수: {file_num}")
    print("클래스별 CSV 매핑:", class_csv_mapping)
    
    for index in tqdm(range(csv_column)):    
        current_class = csv_data[0].iloc[index]['class']
        try:
            if current_class in CLASS2IND:
                current_class_index = CLASS2IND[current_class]
            elif '_' in current_class:
                current_class_index = int(current_class.split('_')[1])
            else:
                digits = ''.join(filter(str.isdigit, current_class))
                if not digits:
                    raise ValueError(f"클래스 이름에서 숫자를 찾을 수 없습니다: {current_class}")
                current_class_index = int(digits)
        except Exception as e:
            print(f"클래스 번호 추출 실패: {current_class}, 에러: {e}")
            current_class_index = 0
        
        # 현재 클래스에 해당하는 CSV 인덱스 가져오기 (기본값 0)
        csv_index = class_csv_mapping.get(current_class_index, 0)
        if csv_index >= file_num:
            print(f"경고: 클래스 {current_class_index}의 CSV 인덱스({csv_index})가 유효하지 않습니다. 기본값 0을 사용합니다.")
            csv_index = 0
        
        data = csv_data[csv_index]
        if(type(data.iloc[index]['rle']) == float):
            result_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
        else:
            result_image = decode_rle_to_mask(data.iloc[index]['rle'], IMAGE_HEIGHT, IMAGE_WIDTH)

        rles.append(encode_mask_to_rle(result_image))
        filename_and_class.append(f"{current_class}_{csv_data[0].iloc[index]['image_name']}")

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    # 기본 Dataframe의 구조는 image_name, class, rle로 되어있습니다.
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    # 최종 ensemble output 저장
    df.to_csv(save_dir, index=False)

if __name__ == "__main__":
    # 후보 모델 경로 설정 (3개의 CSV 파일 예시)
    csv_paths = [
        '../prediction/b3_95_submission.csv',  # index 0
        '../prediction/submission05095.csv',  # index 1
        '../prediction/DLVP_50.csv'   # index 2
    ]
    # 클래스별 CSV 매핑 설정
    # key: 클래스 번호, value: 사용할 CSV 파일의 인덱스
    class_csv_mapping = {
        2: 2,   # 클래스 3은 두 번째 CSV 사용
        6: 1,   # 클래스 7은 세 번째 CSV 사용
        10: 1,  # 클래스 11은 첫 번째 CSV 사용
        14: 1,
        20: 1,
        21: 1,
        24: 1,
        26: 1,
        27: 1
    }
    # 매핑되지 않은 클래스는 기본적으로 첫 번째 CSV(인덱스 0) 사용

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
    save_dir = "ensemble_multi_csv_result.csv"
    csv_ensemble(csv_paths, save_dir, class_csv_mapping, CLASS2IND)