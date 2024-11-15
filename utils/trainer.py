import os
import datetime
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils.dataset import CLASSES
from utils.method import dice_coef


# Wandb import(Feature:#3 Wandb logging, deamin, 2024.11.12)
import wandb
# matplotlib import(#3 commit, deamin, 2024.11.12)
import matplotlib.pyplot as plt
# plotly import(Feature:#6 Wandb에 전체 class 별 dice 시각화 개선, deamin, 2024.11.13)
import plotly.graph_objects as go

def train(model, data_loader, val_loader, criterion, optimizer, num_epochs, val_every, saved_dir, model_name, wandb=None):
    print('Start training..')
    
    best_dice = 0.
    best_epoch = 0
    # 클래스별 다이스 스코어 히스토리 저장을 위한 리스트 추가
    dice_history = {class_name: [] for class_name in CLASSES}
    epoch_history = []
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Training: Epoch [{epoch+1}/{num_epochs}]")
        for step, (images, masks) in progress_bar:
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            outputs = model(images)['out']
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
                })
            
        # 검증 주기마다 검증 수행
        if (epoch + 1) % val_every == 0:
            dice, dices_per_class = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {saved_dir}")
                if best_dice > 0:
                    del_model(f"{model_name}_epoch_{best_epoch}_dice_{best_dice:.4f}",saved_dir)
                best_dice = dice
                best_epoch = epoch + 1
                model_path = save_model(model, f"{model_name}_epoch_{epoch + 1}_dice_{dice:.4f}", saved_dir)
                # Best 모델 파일 저장
                wandb.save(model_path)
                
            # Wandb에 검증 지표와 모델 파일 기록
            # 히스토리 업데이트
            epoch_history.append(epoch)
            for class_name, class_dice in zip(CLASSES, dices_per_class):
                    dice_history[class_name].append(class_dice.item())
                
            # 기본 메트릭 로깅
            wandb.log({
                "valid/mean_dice": dice,
                "valid/epoch": epoch,
            })
            
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

            
                        
def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Validation [{epoch}]")
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in progress_bar:
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
            progress_bar.set_postfix(loss=round(loss.item(), 4), avg_loss=round(total_loss.item() / cnt, 4))

                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice, dices_per_class  # class별 dice도 함께 반환

def save_model(model, model_name, saved_dir):
    output_path = os.path.join(saved_dir, f"{model_name}.pt")
    torch.save(model, output_path)
    return output_path  # 저장된 모델 경로 반환

def del_model(model_name, saved_dir):
    prev_path = os.path.join(saved_dir, f"{model_name}.pt")
    if os.path.exists(prev_path):
        os.remove(prev_path) 

def set_seed(seed=21):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)