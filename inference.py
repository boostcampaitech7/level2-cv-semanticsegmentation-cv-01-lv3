import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
import argparse
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.method import encode_mask_to_rle
from utils.dataset import IND2CLASS, XRayInferenceDataset


def apply_cca(mask, min_size=500, max_components=3):
    """
    Apply more aggressive Connected Component Analysis
    Args:
        mask: Binary mask
        min_size: Minimum component size to keep 
        max_components: Maximum number of components to keep (keep largest ones)
    Returns:
        Cleaned mask
    """
    # Get connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    
    # Create cleaned mask
    cleaned_mask = np.zeros_like(mask)
    
    # Get all component sizes (excluding background)
    sizes = [(label, stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
    
    # Sort components by size (largest first)
    sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Keep only components that meet the size threshold and respect max_components
    count = 0
    for label, size in sizes:
        if size >= min_size and count < max_components:
            cleaned_mask[labels == label] = 1
            count += 1
            
    return cleaned_mask

def test(model, data_loader, thr=0.5, min_component_size=500, max_components=3):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            try:    
                outputs = model(images)['out']
            except:
                outputs = model(images)
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    # Apply more aggressive CCA
                    cleaned_segm = apply_cca(segm, min_size=min_component_size, max_components=max_components)
                    rle = encode_mask_to_rle(cleaned_segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

def parse_args():
    parser = argparse.ArgumentParser(description='X-Ray 이미지 세그멘테이션 추론')
    
    parser.add_argument('--image_root', type=str, default='./data/test/DCM',
                        help='테스트 이미지가 있는 디렉토리 경로')
    parser.add_argument('--model_path', type=str, default='/data/ephemeral/home/kenlee/level2-cv-semanticsegmentation-cv-01-lv3/checkpoints/DELETE_11-19_07-56-38_UPerNet(resnest101e)/UPerNet_Exp_Loss_epoch_196_dice_0.9599.pt',
                        help='학습된 모델 파일 경로')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='배치 크기')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='세그멘테이션 임계값')
    parser.add_argument('--output_path', type=str, default='DICE_Loss_applied_cca_2000.csv',
                        help='결과 저장할 CSV 파일 경로')
    parser.add_argument('--img_size', type=int, default=512,
                        help='입력 이미지 크기')
    parser.add_argument('--min_component_size', type=int, default=2000,
                        help='Minimum size for connected components')
    parser.add_argument('--max_components', type=int, default=1,
                        help='Maximum number of components to keep')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 모델 로드
    model = torch.load(args.model_path)
    
    # 데이터셋 및 데이터로더 설정
    tf = A.Compose([
        A.Resize(args.img_size, args.img_size),
    ])
    
    test_dataset = XRayInferenceDataset(args.image_root, transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    # 추론 수행
    rles, filename_and_class = test(
        model, 
        test_loader, 
        thr=args.threshold,
        min_component_size=args.min_component_size,
        max_components=args.max_components
    )
    
    # submission 파일 생성
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()