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

def train(model, data_loader, val_loader, criterion, optimizer, num_epochs, val_every, saved_dir, model_name, wandb=None):
    print('Start training..')
    
    best_dice = 0.
    for epoch in range(num_epochs):
        model.train()
        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
            
            # Wandb에 학습 지표 기록
            if wandb is not None:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": epoch * len(data_loader) + step,
                    "train/epoch": epoch,
                })
            
        # 검증 주기마다 검증 수행
        if (epoch + 1) % val_every == 0:
            dice, dices_per_class = validation(epoch + 1, model, val_loader, criterion)  # validation 함수가 class별 dice도 반환하도록 수정
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {saved_dir}")
                best_dice = dice
                model_path = save_model(model, f"{model_name}.pt", saved_dir)
                
                # Wandb에 검증 지표와 모델 파일 기록
                if wandb is not None:
                    # 전체 Dice score 기록
                    wandb.log({
                        "valid/mean_dice": dice,
                        "valid/epoch": epoch,
                    })
                    
                    # 클래스별 Dice score 기록
                    for class_name, class_dice in zip(CLASSES, dices_per_class):
                        wandb.log({
                            f"valid/dice_{class_name}": class_dice.item(),
                            "valid/epoch": epoch
                        })
                    
                    # Best 모델 파일 저장
                    wandb.save(model_path)
                    
                    # 클래스별 Dice score를 하나의 그래프로 시각화
                    wandb.log({
                        "valid/class_wise_dice": wandb.plot.line_series(
                            xs=[epoch] * len(CLASSES),
                            ys=[[d.item() for d in dices_per_class]],
                            keys=CLASSES,
                            title="Class-wise Dice Scores",
                            xname="epoch"
                        )
                    })

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
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

def set_seed(seed=21):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)