import torch
import numpy as np

## Dice Coefficient
def dice_coef(y_true, y_pred): # B, C,H*W
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps) 

## RLE 
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width) 

### (11.21) multi-label confusion metrix  
def calculate_confusion_matrix(y_true, y_pred, n_classes):
    """
    y_true, y_pred: shape (B, C, H, W)
    returns: normalized confusion matrix (C, C)
    """
    # 텐서를 (B, C, H*W) 형태로 변환
    y_true_f = y_true.flatten(2)  
    y_pred_f = y_pred.flatten(2)  
    
    # 배치 크기
    batch_size = y_true_f.size(0)
   
    # confusion matrix 초기화 (n_classes x n_classes)
    conf_matrix = torch.zeros((n_classes, n_classes), device=y_true.device)
    
    # 배치의 각 이미지에 대해 계산
    for b in range(batch_size):
        for i in range(n_classes):  # ground truth class (행)
            gt_area = torch.sum(y_true_f[b, i] , -1) + 1e-6  # 분모가 0이 되는 것을 방지
            #초기 gt_area
            if gt_area < 0 : 
                continue
            for j in range(n_classes):  # predicted class (열)
                # intersection 계산
                intersection = torch.sum(y_true_f[b, i] * y_pred_f[b, j] , -1)
                # confusion matrix 업데이트
                # (i,j): gt가 i클래스일 때 j클래스로 예측한 비율
                conf_matrix[i, j] += (intersection / gt_area)
    conf_matrix = conf_matrix/batch_size
    return conf_matrix