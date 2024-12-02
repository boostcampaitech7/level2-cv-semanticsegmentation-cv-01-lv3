import json
import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadXRayAnnotations(BaseTransform):
    def transform(self, results):
        label_path = results["seg_map_path"]
        image_size = (2048, 2048)
        
        # process a label of shape (H, W, NC)
        label_shape = image_size + (29,)  # 29 classes
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = self.CLASSES.index(c)
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image_size, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
            
        results["gt_seg_map"] = label
        return results

@TRANSFORMS.register_module()
class TransposeAnnotations(BaseTransform):
    def transform(self, results):
        results["gt_seg_map"] = np.transpose(results["gt_seg_map"], (2, 0, 1))
        return results 