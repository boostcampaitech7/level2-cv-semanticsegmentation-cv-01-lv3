import os
import sys
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F

from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings('ignore')

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

class EnsembleDataset(Dataset):
    """
    Soft Voting을 위한 DataSet 클래스입니다. 이 클래스는 이미지를 로드하고 전처리하는 작업과
    구성 파일에서 지정된 변환을 적용하는 역할을 수행합니다.

    Args:
        fnames (set) : 로드할 이미지 파일 이름들의 set
        cfg (dict) : 이미지 루트 및 클래스 레이블 등 설정을 포함한 구성 객체
        tf_dict (dict) : 이미지에 적용할 Resize 변환들의 dict
    """    
    def __init__(self, fnames, cfg, transforms_dict):
        self.fnames = sorted(list(fnames))
        self.cfg = cfg
        self.transforms_dict = transforms_dict
        self.models_info = cfg.model_paths
        
        # 클래스 이름과 인덱스 매핑
        self.ind2class = {i : v for i, v in enumerate(cfg.CLASSES)}
        
    def __len__(self):
        return len(self.fnames)
    
    def _load_image(self, fname):
        """이미지를 로드하는 함수"""
        image_path = os.path.join(self.cfg.image_root, fname)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def __getitem__(self, index):
        """
        Args:
            index (int): 데이터 인덱스
        Returns:
            dict: 'image'와 'image_name'을 포함하는 dictionary
        """
        image_name = self.fnames[index]
        image = self._load_image(image_name)
        
        return {
            'image': image,
            'image_name': image_name
        }

    def collate_fn(self, batch):
        """
        배치 데이터를 처리하는 커스텀 collate 함수
        """
        images = [data['image'] for data in batch]
        image_names = [data['image_name'] for data in batch]
        inputs = {"images": images}
        
        # 각 모델별로 transform 적용
        image_dict = self._apply_transforms(inputs)
        
        # numpy array를 torch tensor로 변환 (NHWC -> NCHW)
        image_dict = {k: torch.from_numpy(v.transpose(0, 3, 1, 2)).float() 
                     for k, v in image_dict.items()}#
        
        # 이미지 크기 검증
        for model_name, model_info in self.models_info.items():
            if model_name in image_dict:
                image_batch = image_dict[model_name]
                assert len(image_batch.shape) == 4, \
                    f"collate_fn 내부에서 image_batch의 차원은 반드시 4차원이어야 합니다.\n 현재 shape : {image_batch.shape}"
                
                # 이미지 크기 확인
                expected_size = model_info.get('size', 1024)
                assert image_batch.shape == (len(batch), 3, expected_size, expected_size), \
                    f"collate_fn 내부에서 image_batch의 shape은 ({len(batch)}, 3, {expected_size}, {expected_size})이어야 합니다.\n 현재 shape : {image_batch.shape}"

        return image_dict, image_names
    
    def _apply_transforms(self, inputs):
        """각 모델별로 transform을 적용하는 함수"""
        images = inputs["images"]
        transformed_dict = {}
        
        for model_name, model_info in self.models_info.items():
            transformed_images = []
            for image in images:
                # 해당 모델의 transform 가져오기
                transform = self.transforms_dict[model_info.path]
                # transform 적용
                transformed = transform(image=image)["image"]
                transformed_images.append(transformed)
            
            transformed_dict[model_name] = np.stack(transformed_images)
        
        return transformed_dict


def encode_mask_to_rle(mask):
    # mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_models(cfg, device):
    """
    구성 파일에 지정된 경로에서 모델을 로드합니다.

    Args:
        cfg (dict): 모델 경로가 포함된 설정 객체
        device (torch.device): 모델을 로드할 장치 (CPU or GPU)

    Returns:
        dict: 처리 이미지 크기별로 모델을 그룹화한 dict
        int: 로드된 모델의 총 개수
    """    
    model_dict = {}
    model_count = 0

    print("\n======== Model Load ========")
    # inference 해야하는 "이미지 크기 별" -> 개별로로 모델 순차저장
    for  models_name in cfg.model_paths.keys(): # model_names

        model_dict[models_name] = []
        model_info = cfg.model_paths[models_name]
        print(f"{model_info.get('size', 1024)} image size 추론 모델 불러오기 진행 시작")
        #image_size
        path = model_info['path']  # 'path' 키에서 경로를 가져옵니다.
        print(f"{osp.basename(path)} 모델을 불러오는 중입니다..", end="\t")

        #예외처리 :(1) torch.save(model) (2) torch.save(model.state_dict())           +
        checkpoint = torch.load(path)
        print(type(checkpoint))
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint: 
            print("state_dict 형식으로 저장된 모델입니다. 모델 클래스를 초기화합니다.")
            
            # DeepLabV3Plus 초기화 (필요시 다른 모델 클래스 사용 가능)
            model_instance = smp.DeepLabV3Plus(
                encoder_name="efficientnet-b7",
                encoder_weights="imagenet",
                in_channels=3,
                classes=len(cfg.CLASSES),
                aux_params={
                    "classes": len(cfg.CLASSES),
                    "dropout": 0.5,
                },
                decoder_channels=256,
                encoder_output_stride=16,
                decoder_atrous_rates=(4, 8, 12),
            )

            model_instance.load_state_dict(checkpoint['model_state_dict'])
            model_instance = model_instance.cuda()
        else : 
            model_instance = checkpoint
            model_instance = model_instance.to(device)
        #모 load완료
        # model_instance = model_instance.to(device)
        model_instance.eval()
        model_dict[models_name].append(model_instance)
        model_count += 1
        print("불러오기 성공!")
        print()

    print(f"모델 총 {model_count}개 불러오기 성공!\n")
    return model_dict, model_count # {img_size -> models_name : torch.load(model) } , model_count



def save_results(cfg, filename_and_class, rles):
    """
    추론 결를 csv 파일로 저장합니다.

    Args:
        cfg (dict): 출력 설정을 포함하는 구성 객체
        filename_and_class (list): 파일 이름과 클래스 레이블이 포함된 list
        rles (list): RLE로 인코딩된 세크멘테이션 마스크들을 가진 list
    """    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    print("\n======== Save Output ========")
    print(f"{cfg.save_dir} 폴더 내부에 {cfg.output_name}을 생성합니다..", end="\t")
    os.makedirs(cfg.save_dir, exist_ok=True)

    output_path = osp.join(cfg.save_dir, cfg.output_name)
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"{output_path}를 생성하는데 실패하였습니다.. : {e}")
        raise

    print(f"{osp.join(cfg.save_dir, cfg.output_name)} 생성 완료")



def soft_voting(cfg):
    """
    Soft Voting을 수행합니다. 여러 모델의 예측을 결합하여 최종 예측을 생성
    """    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 이미지 파일 목록 가져오기
    fnames = {
        osp.relpath(osp.join(root, fname), start=cfg.image_root)
        for root, _, files in os.walk(cfg.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }
    
    # transforms dictionary 생성
    tf_dict = {}
    for model_name, model_info in cfg.model_paths.items():
        transforms = []
        for transform in model_info.get('transforms', []):
            if transform['type'] == 'CLAHE':
                if transform['apply']:
                    transforms.append(A.CLAHE(
                        clip_limit=transform.get('clip_limit', 4.0),
                        tile_grid_size=transform.get('tile_grid_size', [3, 3]),
                        p=transform.get('p', 1.0)
                    ))
            elif transform['type'] == 'Resize':
                transforms.append(A.Resize(height=transform['height'], 
                                        width=transform['width']))
        tf_dict[model_info['path']] = A.Compose(transforms)
    
    print("\n======== PipeLine 생성 ========")
    for k, v in tf_dict.items():
        print(f"{osp.basename(k)} 모델은 {v} pipeline으로 처리됩니다.")
    
    # 데이터셋과 데이터로더 생성
    dataset = EnsembleDataset(fnames, cfg, tf_dict)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=dataset.collate_fn
    )
    
    model_dict, model_count = load_models(cfg, device)
    
    filename_and_class = []
    rles = []
    min_component_size = 2000
    max_components = 1

    print("======== Soft Voting Start ========")
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Inference...]", disable=False) as pbar:
            print(f"data_loader : {len(data_loader)} ")
            for image_dict, image_names in data_loader:
                total_output = torch.zeros((cfg.batch_size, len(cfg.CLASSES), 2048, 2048)).to(device)
                # print(f"model_dict : {model_dict}")
                
                for model_name, models_list in model_dict.items():
                    print(f"model_name : {model_dict[model_name]}")
                    for model in models_list:
                        outputs = model(image_dict[model_name].to(device))
                        # print("#", outputs.shape) # torch.Size([2, 29, 1024, 1024]) , torch.Size([2, 29, 1536, 1536])
                        # if isinstance(outputs, tuple):
                        #     outputs = outputs[0]
                        
                        print("##",torch.unique(outputs))
                        outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                        outputs = torch.sigmoid(outputs)
                        print("###",torch.unique(outputs))
                        total_output += outputs
                
                total_output /= model_count
                total_output = (total_output > cfg.threshold).detach().cpu().numpy()
                

                for output, image_name in zip(total_output, image_names):
                    for c, segm in enumerate(output):
                        # cleaned_segm = apply_cca(segm, min_size=min_component_size, max_components=max_components)
                        # print("#",len(segm) , segm.shape, torch.unique(segm) ) # 2048 ,(2048,2048)
                        # print("####")
                        rle = encode_mask_to_rle(segm)
                        # print("##",len(rle)) # 0
                        rles.append(rle)
                        filename_and_class.append(f"{dataset.ind2class[c]}_{image_name}")
                
                pbar.update(1)

    save_results(cfg, filename_and_class, rles)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="utils/soft_voting_setting.yaml")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)

    if cfg.root_path not in sys.path:
        sys.path.append(cfg.root_path)
    
    soft_voting(cfg)