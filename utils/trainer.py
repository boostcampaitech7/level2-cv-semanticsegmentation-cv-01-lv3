import os
import datetime
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils.dataset import CLASSES
#Calculate Multi-label confusion metrix (2024.11.21)
from utils.method import dice_coef , calculate_confusion_matrix 
# Wandb import(Feature:#3 Wandb logging, deamin, 2024.11.12)
import wandb
# matplotlib import(#3 commit, deamin, 2024.11.12)
import matplotlib.pyplot as plt
# plotly import(Feature:#6 Wandb에 전체 class 별 dice 시각화 개선, deamin, 2024.11.13)
import plotly.graph_objects as go
import seaborn as sns

def train(model, data_loader, val_loader, criterion, optimizer, scheduler, num_epochs, val_every, saved_dir, model_name, early_stopping=True, patience=5, wandb=None):
    print('Start training..')
    if early_stopping:
        print(f'Early stopping enabled with patience {patience}')
    
    best_dice = 0.
    best_loss = float('inf')  # 최소 loss 초기화
    best_epoch = 0
    # Early stopping 관련 변수
    early_stopping_counter = 0
    best_model_state = None
    
    # 클래스별 다이스 스코어 히스토리 저장을 위한 리스트
    dice_history = {class_name: [] for class_name in CLASSES}
    epoch_history = []
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Training: Epoch [{epoch+1}/{num_epochs}]")
        for step, (images, masks) in progress_bar:
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            try:
                outputs = model(images)['out']
            except:
                outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            progress_bar.set_postfix(loss=round(loss.item(), 4), time=datetime.datetime.now().strftime("%H:%M:%S"))
                            
            # Wandb에 학습 지표 기록
            if wandb is not None:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": epoch * len(data_loader) + step,
                    "train/epoch": epoch,
                    "train/learning_rate": scheduler.get_last_lr()[0]  # 현재 학습률 로깅
                })
        
        # 에폭 끝날 때마다 scheduler step
        scheduler.step()
        
        # 검증 주기마다 검증 수행
        if (epoch + 1) % val_every == 0:
            val_loss, dice, dices_per_class , norm_confusion_matrix = validation(epoch + 1, model, val_loader, criterion)
            
            # Early stopping 체크 (loss 기준)
            if early_stopping:
                if val_loss < best_loss:
                    print(f"Best validation loss at epoch: {epoch + 1}, {best_loss:.4f} -> {val_loss:.4f}")
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    print(f'Early stopping counter: {early_stopping_counter}/{patience}')
                    
                    if early_stopping_counter >= patience:
                        print(f'Early stopping triggered at epoch {epoch + 1}')
                        # 최고 성능 모델로 복원
                        model.load_state_dict(best_model_state)
                        if wandb:
                            wandb.finish()
                        return
            
            # Best dice 모델 저장 
            if dice > best_dice:
                print(f"Best dice score at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {saved_dir}")
                if best_dice > 0:
                    del_model(f"{model_name}_epoch_{best_epoch}_dice_{best_dice:.4f}", saved_dir)
                best_dice = dice
                best_epoch = epoch + 1
                model_path = save_model(model, f"{model_name}_epoch_{epoch + 1}_dice_{dice:.4f}", saved_dir)
                wandb.save(model_path)
            
            # Wandb logging
            wandb.log({
                # validation loss 추가
                "valid/loss": val_loss,
                "valid/mean_dice": dice,
                "valid/epoch": epoch,
            })
            
            # 히스토리 업데이트
            epoch_history.append(epoch)
            for class_name, class_dice in zip(CLASSES, dices_per_class):
                    dice_history[class_name].append(class_dice.item())
                
            # 클래스별 Dice score 기록
            class_dice_dict = {}
            for class_name, class_dice in zip(CLASSES, dices_per_class):
                class_dice_dict[f"valid/dice_{class_name}"] = class_dice.item()
            wandb.log(class_dice_dict)

           
            # 데이터가 충분히 쌓였을 때만 그래프 그리기
            if len(epoch_history) > 0:

                fig = go.Figure()
                # 색상 팔레트 생성
                colors = plt.cm.rainbow(np.linspace(0, 1, len(CLASSES)))
                
                # 각 클래스별 라인 추가
                for idx, (class_name, color) in enumerate(zip(CLASSES, colors)):
                    # RGB 색상을 hex 코드로 변환
                    hex_color = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                    
                    # 라인 스타일 설정
                    dash_style = 'solid' if idx % 2 == 0 else 'dash'
                    
                    fig.add_trace(go.Scatter(
                        x=epoch_history,
                        y=dice_history[class_name],
                        name=class_name,
                        line=dict(color=hex_color, dash=dash_style, width=2),
                        mode='lines+markers',
                        marker=dict(
                            symbol=['circle', 'square', 'diamond', 'triangle-up', 'cross'][idx % 5],
                            size=6
                        )
                    ))
                
                # 레이아웃 설정
                fig.update_layout(
                    title='Class-wise Dice Scores Over Time',
                    xaxis_title='Epoch',
                    yaxis_title='Dice Score',
                    legend=dict(
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.05,
                        font=dict(size=10)
                    ),
                    showlegend=True,
                    width=1000,
                    height=600,
                    plot_bgcolor='white',
                    hovermode='x unified'
                )
                
                # 그리드 추가
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                
                # wandb에 로깅
                wandb.log({"valid/class_wise_dice_scores": fig})

                ## logging confusion_matrix in wandb
                conf_fig = go.Figure(data=go.Heatmap(
                    z=norm_confusion_matrix.cpu().numpy(),
                    x=CLASSES,
                    y=CLASSES,
                    hoverongaps=False,
                    text=norm_confusion_matrix.cpu().numpy(),
                    texttemplate="%{text:.4f}",
                    textfont={"size": 10},
                    colorscale="OrRd",
                    colorbar=dict(title="Percentage"),
                ))
                conf_fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted Class",
                    yaxis_title="Actual Class",
                    width=800,
                    height=800,
                )

                wandb.log({
                    "valid/confusion_matrix": conf_fig
                })
            
                        
def visualize_mismatch(worst_batch_data, epoch, num_visualize, data_loader_length):
    """Visualize the worst performing images with their predictions and misclassification maps"""
    if worst_batch_data is None or not wandb:
        return
        
    n_vis = min(worst_batch_data['images'].size(0), num_visualize)
    dice_scores = worst_batch_data['dice_scores'].mean(dim=1)
    sorted_indices = torch.argsort(dice_scores)
    
    mismatch_images = []  # 이미지들을 리스트로 모음
    
    for idx in range(n_vis):
        img_idx = sorted_indices[idx]
        sample_image = worst_batch_data['images'][img_idx]
        sample_mask = worst_batch_data['masks'][img_idx]
        sample_pred = worst_batch_data['predictions'][img_idx]
        sample_dice = dice_scores[img_idx].item()
        
        # 오분류 맵 생성
        misclass_map = torch.zeros(3, sample_mask.size(1), sample_mask.size(2))
        for cls_idx in range(len(CLASSES)):
            misclass_map[0] += ((sample_pred[cls_idx] == 1) & (sample_mask[cls_idx] == 0)).float()  # FP (Red)
            misclass_map[2] += ((sample_pred[cls_idx] == 0) & (sample_mask[cls_idx] == 1)).float()  # FN (Blue)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 원본 이미지
        axes[0].imshow(sample_image.permute(1,2,0))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground Truth
        gt_vis = sample_mask.sum(0)
        axes[1].imshow(gt_vis, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # 예측 결과
        pred_vis = sample_pred.sum(0)
        axes[2].imshow(pred_vis, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # 오분류 맵
        axes[3].imshow(misclass_map.permute(1,2,0))
        axes[3].set_title(f'Misclassification Map\nDice Score: {sample_dice:.4f}')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # 이미지를 리스트에 추가
        mismatch_images.append(wandb.Image(fig))
        plt.close()
    
    # 한번에 모든 이미지를 Media로 로깅
    wandb.log({
        "Media/Mismatch_Visualization": mismatch_images,
        "epoch": epoch,
        "step": epoch * data_loader_length
    })

def validation(epoch, model, data_loader, criterion, thr=0.5, num_visualize=4):
    print(f'Start validation #{epoch:2d}')
    model.eval()
    
    dices = []
    total_loss = 0
    cnt = 0
    worst_batch_dice = float('inf')
    worst_batch_data = None
    n_classes = len(CLASSES)
    total_conf_matrix = torch.zeros((n_classes, n_classes), device='cuda')
    
    misclassification_info = {cls: {'false_positives': 0, 'false_negatives': 0} for cls in CLASSES}
    total_pixels = 0
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Validation [{epoch}]")
        n_classes = len(CLASSES)
        total_conf_matrix = torch.zeros((n_classes, n_classes), device='cuda')

        for step, (images, masks) in progress_bar:
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            try:
                outputs = model(images)['out']
            except:
                outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
    
            loss = criterion(outputs, masks)
            total_loss += loss.item() # gradient 계산이 필요 없으므로 item() 사용
            cnt += 1

            outputs = torch.sigmoid(outputs)
            predictions = (outputs > thr).float()
            
            # Dice score 계산
            current_dice = dice_coef(predictions, masks)
            dices.append(current_dice)
            batch_avg_dice = torch.mean(current_dice).item()
            progress_bar.set_postfix(loss=round(loss.item(), 4), avg_loss=round(total_loss / cnt, 4))
            # Worst batch 저장
            if batch_avg_dice < worst_batch_dice:
                worst_batch_dice = batch_avg_dice
                worst_batch_data = {
                    'images': images.cpu(),
                    'masks': masks.cpu(),
                    'predictions': predictions.cpu(),
                    'dice_scores': current_dice
                }
            
            # Confusion matrix 업데이트
            total_conf_matrix += calculate_confusion_matrix(masks, predictions, n_classes)
            
            # 오분류 정보 수집
            total_pixels += images.size(0) * images.size(2) * images.size(3)
            for cls_idx, cls_name in enumerate(CLASSES):
                false_positives = ((predictions[:, cls_idx] == 1) & (masks[:, cls_idx] == 0)).sum().item()
                false_negatives = ((predictions[:, cls_idx] == 0) & (masks[:, cls_idx] == 1)).sum().item()
                
                misclassification_info[cls_name]['false_positives'] += false_positives
                misclassification_info[cls_name]['false_negatives'] += false_negatives
            
    
    # Mismatch visualization 호출 시 data_loader 길이도 전달
    visualize_mismatch(worst_batch_data, epoch, num_visualize, len(data_loader))
    
    # 오분류 통계 로깅
    if total_pixels > 0 and wandb:
        misclass_stats = {}
        for cls_name in CLASSES:
            fp = misclassification_info[cls_name]['false_positives']
            fn = misclassification_info[cls_name]['false_negatives']
            fp_rate = fp / total_pixels * 100
            fn_rate = fn / total_pixels * 100
            misclass_stats.update({
                f"Mismatch Visualization/{cls_name}_FP_rate": fp_rate,
                f"Mismatch Visualization/{cls_name}_FN_rate": fn_rate
            })
        wandb.log(misclass_stats)
    
    # 최종 메트릭 계산
    avg_conf_matrix = total_conf_matrix / cnt
    avg_loss = total_loss / cnt
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    avg_dice = torch.mean(dices_per_class).item()
    
    ##(11.21) Confusion Metrix heatmap 추가
    plt.figure(figsize=(20, 16))
    # GPU 텐서를 CPU로 이동 후 numpy로 변환
    conf_matrix_np = avg_conf_matrix.cpu().numpy()
    
    sns.heatmap(conf_matrix_np, 
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                xticklabels=CLASSES,
                yticklabels=CLASSES)
    
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    try:
        # wandb에 confusion matrix 이미지 로깅
        if wandb.run is not None:
            wandb.log({
                "confusion_matrix": wandb.Image(plt),
                "epoch": epoch
            })
            print(f"Saved {epoch} - Confusion matrix")
    except Exception as e:
        print(f"Wandb logging error: {str(e)}")
    finally:
        plt.close()  # 메모리 해제
    
    return avg_loss, avg_dice, dices_per_class, avg_conf_matrix

def save_model(model, model_name, saved_dir):
    output_path = os.path.join(saved_dir, f"{model_name}.pt")
    torch.save(model, output_path)
    return output_path  # 저장된 모델 경로 반환

def del_model(model_name, saved_dir):
    prev_path = os.path.join(saved_dir, f"{model_name}.pt")
    if os.path.exists(prev_path):
        os.remove(prev_path) 

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

