from typing import Optional

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

@METRICS.register_module()
class DiceMetric(BaseMetric):
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    @staticmethod
    def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten(-2)
        y_pred_f = y_pred.flatten(-2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)
        
        eps = 0.0001
        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data']
            label = data_sample['gt_sem_seg']['data'].to(pred_label)
            self.results.append(
                self.dice_coef(label, pred_label)
            )

    def compute_metrics(self, results):
        results = torch.stack(self.results, 0)
        dices_per_class = torch.mean(results, 0)
        
        metrics = {
            "mDice": torch.mean(dices_per_class).item()
        }
        
        return metrics 