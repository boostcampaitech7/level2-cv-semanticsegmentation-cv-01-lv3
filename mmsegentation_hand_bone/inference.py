from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
import torch
import os
import pandas as pd
from tqdm import tqdm
import argparse
from utils.method import encode_mask_to_rle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

def visualize_first_image(img, pred_mask, save_path='first_result.png'):
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # 세그멘테이션 맵
    combined_mask = np.zeros((*pred_mask.shape[1:], 3), dtype=np.float32)
    colors = plt.cm.rainbow(np.linspace(0, 1, pred_mask.shape[0]))[:, :3]
    
    for i in range(pred_mask.shape[0]):
        for j in range(3):
            combined_mask[..., j] += pred_mask[i] * colors[i, j]
    
    combined_mask = np.clip(combined_mask, 0, 1)
    
    plt.subplot(1, 3, 2)
    plt.imshow(combined_mask)
    plt.title('Segmentation Map')
    plt.axis('off')
    
    # 오버레이
    img_normalized = img.astype(np.float32) / 255.0
    if len(img_normalized.shape) == 2:
        img_normalized = np.stack([img_normalized] * 3, axis=-1)
    overlay = img_normalized * 0.7 + combined_mask * 0.3
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test(model, image_paths, thr=0.6):
    model.eval()
    results = {}
    first_image = True
    
    with torch.no_grad():
        for image_path in tqdm(image_paths):
            # 이미지 로드
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # # 추론 수행
            # result = inference_model(model, img)
            # pred_logits = result.pred_sem_seg.data
            # print("Raw logits range:", pred_logits.min().item(), pred_logits.max().item())  # 추가
            
            # pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
            # print("After sigmoid range:", pred_probs.min(), pred_probs.max())  # 추가

            # # pred_mask = result.pred_sem_seg.data.cpu().numpy()
            # # Sigmoid 적용 전 원본 logits 가져오기
            # pred_logits = result.pred_sem_seg.data
            # # Sigmoid 적용하여 확률값 얻기 (0~1 사이 값)
            # pred_probs = torch.sigmoid(pred_logits).cpu().numpy() 
            # 임계값 적용하여 이진 마스크 생성
            result = inference_model(model, img)
            pred_probs = result.pred_sem_seg.data.cpu().numpy()  # sigmoid 제거
            print("Probability range:", pred_probs.min(), pred_probs.max())
            
            pred_mask = (pred_probs > thr)

            # 첫 번째 이미지에 대해서만 시각화
            if first_image:
                # print("Probability range:", pred_probs.min(), pred_probs.max())
                # print("Probability histogram:", np.histogram(pred_probs, bins=29))
                visualize_first_image(img, pred_mask)
                first_image = False
            
            # RLE 인코딩
            image_name = os.path.basename(image_path)
            results[image_name] = {CLASSES[i]: '' for i in range(len(CLASSES))}
            
            for c in range(len(CLASSES)):
                rle = encode_mask_to_rle(pred_mask[c])
                if rle:
                    results[image_name][CLASSES[c]] = rle
    
    # 결과를 리스트로 변환
    rles = []
    filename_and_class = []
    
    for image_name in sorted(results.keys()):
        for class_name in sorted(CLASSES):
            rles.append(results[image_name][class_name])
            filename_and_class.append(f"{class_name}_{image_name}")
    
    return rles, filename_and_class

def parse_args():
    parser = argparse.ArgumentParser(description='X-Ray 이미지 세그멘테이션 추론')
    parser.add_argument('--config', type=str, 
                       default='configs/segformer/segformer_mit-b5_xray.py',
                       help='모델 설정 파일 경로')
    parser.add_argument('--checkpoint', type=str, 
                       default='/data/ephemeral/home/deamin/backup/mmsegmentation/work_dirs/segformer_mit-b5_xray_MultiStepLR_1024_20000/best_mDice_iter_13000.pth',
                       help='체크포인트 파일 경로')
    parser.add_argument('--image_root', type=str, 
                       default='/data/ephemeral/home/deamin/backup/level2-cv-semanticsegmentation-cv-01-lv3/data/test/DCM',
                       help='테스트 이미지 디렉토리')
    parser.add_argument('--threshold', type=float, 
                       default=0.5,
                       help='세그멘테이션 임계값')
    parser.add_argument('--output_path', type=str, 
                       default='b5_70_submission.csv',
                       help='결과 저장 CSV 파일 경로')
    return parser.parse_args()

def get_image_paths(root_dir):
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                image_paths.append(os.path.join(dirpath, filename))
    return sorted(image_paths)

def main():
    args = parse_args()
    register_all_modules()
    
    # 모델 초기화
    model = init_model(args.config, args.checkpoint, device='cuda:0')
    
    # 이미지 경로 리스트 생성
    image_paths = get_image_paths(args.image_root)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {args.image_root}")
    
    # 추론 수행
    rles, filename_and_class = test(model, image_paths, thr=args.threshold)
    
    # submission 파일 생성
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")

if __name__ == '__main__':
    main()