from numba import njit

@njit
def calculate_dice_score(pred, target):
    """
    개별 배치의 다이스 스코어를 계산하는 함수
    """
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_coef(pred, target):
    """
    배치 전체의 다이스 스코어를 계산하는 래퍼 함수
    """
    batch_size = pred.shape[0]
    num_classes = pred.shape[1]
    dice_scores = np.zeros((batch_size, num_classes))
    
    # numpy로 변환
    pred_np = pred.numpy()
    target_np = target.numpy()
    
    for i in range(batch_size):
        for j in range(num_classes):
            dice_scores[i,j] = calculate_dice_score(pred_np[i,j], target_np[i,j])
            
    return torch.from_numpy(dice_scores) 