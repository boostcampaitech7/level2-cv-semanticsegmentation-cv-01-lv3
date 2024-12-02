import json
import os.path as osp
import cv2
import numpy as np
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class XRayDataset(BaseSegDataset):
    CLASSES = (
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
    )
    
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), 
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176)]

    def __init__(self, is_train, **kwargs):
        self.is_train = is_train
        super().__init__(**kwargs)
        
    def load_data_list(self):
        """Load annotation from directory.
        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', '')
        ann_dir = self.data_prefix.get('seg_map_path', '')
        
        for img_file in os.listdir(img_dir):
            if not img_file.endswith('.png'):
                continue
                
            img_path = osp.join(img_dir, img_file)
            ann_path = osp.join(ann_dir, img_file.replace('.png', '.json'))
            
            data_info = dict(
                img_path=img_path,
                seg_map_path=ann_path
            )
            data_list.append(data_info)
            
        return data_list 