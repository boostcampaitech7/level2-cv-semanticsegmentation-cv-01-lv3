import os.path as osp
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
import numpy as np
import os
import mmengine.fileio as fileio
from sklearn.model_selection import GroupKFold
import random
import torch

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

@DATASETS.register_module()
class XRayDataset(BaseSegDataset):
    def __init__(self, is_train, **kwargs):
        self.is_train = is_train
        super().__init__(**kwargs)

    def load_data_list(self):
        """Load annotation from directory.
        Returns:
            list[dict]: All data info of dataset.
        """
        seed = 1  # 가장 좋은 결과를 보인 시드값
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        img_dir = '/data/ephemeral/home/deamin/backup/level2-cv-semanticsegmentation-cv-01-lv3/data/train/DCM'
        ann_dir = '/data/ephemeral/home/deamin/backup/level2-cv-semanticsegmentation-cv-01-lv3/data/train/outputs_json'
        
        print(f"Checking directories:")
        print(f"Image directory: {img_dir}")
        print(f"Annotation directory: {ann_dir}")
        
        # Get all image files
        _filenames = []
        _labelnames = []
        
        for root, _, files in os.walk(img_dir):
            for fname in files:
                if fname.endswith('.png'):
                    img_path = os.path.join(root, fname)
                    rel_img_path = os.path.relpath(img_path, img_dir)
                    _filenames.append(rel_img_path)
                    
                    # Corresponding label path
                    label_fname = fname.replace('.png', '.json')
                    label_path = os.path.join(ann_dir, os.path.relpath(os.path.join(root, label_fname), img_dir))
                    _labelnames.append(os.path.relpath(label_path, ann_dir))

        _filenames = np.array(_filenames)
        _labelnames = np.array(_labelnames)

        # split train-valid
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for _ in _filenames]

        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue

                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                # skip i > 0
                break

        data_list = []
        for i, (img_path, ann_path) in enumerate(zip(filenames, labelnames)):
            data_info = dict(
                img_path=os.path.join(img_dir, img_path),
                seg_map_path=os.path.join(ann_dir, ann_path),
            )
            data_list.append(data_info)

        return data_list 