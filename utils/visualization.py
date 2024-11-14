import matplotlib.pyplot as plt
import numpy as np
import csv

## Visualization
# 색상 리스트
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image

def visualize_prediction(image, preds):
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image)
    ax[1].imshow(label2rgb(preds))
    plt.show() 

def load_data(csv_path):
    '''
    Predicted된 Segmentation을 이미지 형태로 확인할 수 있도록 하기 위해 output csv파일로부터
    images와 labels를 불러오는 함수입니다.

    params
    csv_path: Output.csv의 경로
    '''
    images = []
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            image_path = row[0]
            label = np.array([int(x) for x in row[1:]]).reshape(-1, 2048, 2048)
            images.append(image_path)
            labels.append(label)
    return images, labels

def save_visualization(image, preds, filename="segmentation_result.png"):
    '''
    Predicted된 Segmentation을 이미지 형태로 저장할 수 있도록 하는 함수입니다.

    params
    image: 원본 이미지
    preds: 예측된 레이블
    filename: 이미지로 저장될 이름
    '''
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(label2rgb(preds))
    ax[1].set_title("Segmentation Mask")
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Image saved as {filename}")