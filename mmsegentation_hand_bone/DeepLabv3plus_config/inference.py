import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.method import encode_mask_to_rle
from utils.dataset import IND2CLASS, XRayInferenceDataset

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    # 결과를 저장할 딕셔너리
    results = {}
    
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            # 각 이미지별로 처리
            for output, image_name in zip(outputs, image_names):
                # 이미지별 결과 딕셔너리 초기화
                if image_name not in results:
                    results[image_name] = {IND2CLASS[i]: '' for i in range(len(IND2CLASS))}
                
                # 각 클래스별 RLE 계산
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    if rle:  # RLE가 존재할 때만 저장
                        results[image_name][IND2CLASS[c]] = rle

    # 결과를 리스트로 변환
    rles = []
    filename_and_class = []
    
    # 정렬된 순서로 결과 추출
    for image_name in sorted(results.keys()):
        for class_name in sorted(IND2CLASS.values()):
            rles.append(results[image_name][class_name])
            filename_and_class.append(f"{class_name}_{image_name}")
                    
    return rles, filename_and_class

def parse_args():
    parser = argparse.ArgumentParser(description='X-Ray 이미지 세그멘테이션 추론')
    
    parser.add_argument('--image_root', type=str, default='./data/test/DCM',
                        help='테스트 이미지가 있는 디렉토리 경로')
    parser.add_argument('--model_path', type=str, default='/data/ephemeral/home/deamin/backup/level2-cv-semanticsegmentation-cv-01-lv3/wandb/run-20241128_104212-dj41f3c7/files/checkpoints/1024_lr1e3_adamW_Fold3.pt',
                        help='학습된 모델 파일 경로')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='배치 크기')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='세그멘테이션 임계값')
    parser.add_argument('--output_path', type=str, default='DLVP_50.csv',
                        help='결과 저장할 CSV 파일 경로')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='입력 이미지 크기')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 모델 로드
    # model = torch.load(args.model_path)
    
    import segmentation_models_pytorch as smp
    from utils.dataset import CLASSES
    
    # aux_params 수정
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(CLASSES),
        aux_params={
            "classes": len(CLASSES),
            "dropout": 0.5
        },
        decoder_channels=256,
        encoder_output_stride=16,
        decoder_atrous_rates=(4, 8, 12)
    )
    # state_dict 로드 및 처리
    checkpoint = torch.load(args.model_path)
    #state_dict = checkpoint['model_state_dict']  # 'model_state_dict' 키로 접근
    model.load_state_dict(checkpoint['model_state_dict'])  # checkpoint에서 model_state_dict 키의 값을 가져옴
    model = model.cuda()

    import albumentations as A
    import cv2
    import numpy as np
    class CustomPreprocessing(A.ImageOnlyTransform):
        def __init__(self, always_apply=False, p=1.0):
            super().__init__(always_apply, p)
            
        def apply(self, image, **params):
            # cv2 처리를 위해 uint8로 변환
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
                
            # grayscale 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Laplacian 적용
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)
            
            # 3채널로 변환하여 addWeighted 적용
            if len(image.shape) == 3:
                laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            
            enhanced = cv2.addWeighted(image.astype(np.float32), 1.0,
                                    laplacian.astype(np.float32), 0.2, 0)
            
            # CLAHE 적용
            enhanced_uint8 = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            if len(enhanced_uint8.shape) == 3:
                # RGB 이미지인 경우 각 채널에 CLAHE 적용
                enhanced = np.zeros_like(enhanced_uint8)
                for i in range(3):
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced[:,:,i] = clahe.apply(enhanced_uint8[:,:,i])
            else:
                # 그레이스케일 이미지인 경우
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(enhanced_uint8)
                
            return enhanced
    # 데이터셋 및 데이터로더 설정
    tf = A.Compose([
        # CustomPreprocessing(p=1),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(3, 3), p=1.0),
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
    rles, filename_and_class = test(model, test_loader, thr=args.threshold)
    
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