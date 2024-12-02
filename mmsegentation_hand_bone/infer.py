from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
import mmcv
import numpy as np
import matplotlib.pyplot as plt

# 모든 모듈 등록
register_all_modules()

# 설정 파일과 체크포인트 파일 경로 지정
config_file = 'configs/segformer/segformer_mit-b3_xray.py'
checkpoint_file = '/data/ephemeral/home/deamin/backup/mmsegmentation/work_dirs/segformer_mit-b5_xray_MultiStepLR_1024_20000/best_mDice_iter_13000.pth'

# 모델 초기화
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 이미지 경로
img_path = '/data/ephemeral/home/deamin/backup/level2-cv-semanticsegmentation-cv-01-lv3/data/test/DCM/ID040/image1661319116107.png'

# 이미지 로드
img = mmcv.imread(img_path)
img = mmcv.imresize(img, (1024, 1024))  # 이미지 리사이즈

# 추론 수행
result = inference_model(model, img)

# 결과 시각화
pred_mask = result.pred_sem_seg.data.cpu().numpy()  # (C, H, W)
num_classes = pred_mask.shape[0]

# 각 클래스별 마스크를 다른 색상으로 표시
plt.figure(figsize=(15, 5))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 전체 세그멘테이션 맵
combined_mask = np.zeros((1024, 1024, 3), dtype=np.float32)
colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))[:, :3]  # RGB 색상 생성

for i in range(num_classes):
    color_mask = colors[i]
    combined_mask += np.dstack([pred_mask[i] * c for c in color_mask])

# 결과 정규화
combined_mask = np.clip(combined_mask, 0, 1)

# 세그멘테이션 맵 표시
plt.subplot(1, 3, 2)
plt.imshow(combined_mask)
plt.title('Segmentation Map')
plt.axis('off')

# 오버레이 이미지
plt.subplot(1, 3, 3)
overlay = img.astype(np.float32) / 255.0 * 0.7 + combined_mask * 0.3
plt.imshow(overlay)
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()
plt.savefig('segmentation_result.png')
plt.close()

print("시각화 결과가 'segmentation_result.png'에 저장되었습니다.")