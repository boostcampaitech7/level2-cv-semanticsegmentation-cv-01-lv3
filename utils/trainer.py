import os
import datetime
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils.dataset import CLASSES
from utils.method import dice_coef

def train(model, data_loader, val_loader, criterion, optimizer, num_epochs, val_every, saved_dir, model_name):
    print('Start training..')
    
    best_dice = 0.
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
                
        if (epoch + 1) % val_every == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {saved_dir}")
                best_dice = dice
                save_model(model, f"{model_name}.pt", saved_dir)

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
    
    return avg_dice



def save_model(model, model_name, saved_dir):
    output_path = os.path.join(saved_dir, f"{model_name}.pt")
    torch.save(model, output_path)

def set_seed(seed=21):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)