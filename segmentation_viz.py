import os
import numpy as np
from PIL import Image
from utils.visualization import PALETTE, save_visualization, load_data
from utils.method import decode_rle_to_mask

def main():
    csv_path = "./output.csv"  # CSV 파일 경로
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 불러오기
    images, labels = load_data(csv_path)

    for i, (image_path, label) in enumerate(zip(images, labels)):
        image = Image.open(image_path)
        image = np.array(image)
        print(image_path)
        # 시각화 결과 저장
        # filename = os.path.join(output_dir, f"segmentation_{i}.png")
        # save_visualization(image, label, filename=filename)

if __name__ == "__main__":
    main()