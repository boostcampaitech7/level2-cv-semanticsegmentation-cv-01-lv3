import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS

@HOOKS.register_module()
class EmptyCacheHook(Hook):
    def __init__(self, after_val=True, after_train_epoch=False):
        self.empty_after_val = after_val  # 변수 이름 변경
        self.empty_after_train_epoch = after_train_epoch  # 변수 이름 변경

    def after_val_epoch(self, runner, metrics=None):
        if self.empty_after_val:
            torch.cuda.empty_cache()

    def after_val(self, runner):  # after_val 메서드 추가
        if self.empty_after_val:
            torch.cuda.empty_cache()

    def after_train_epoch(self, runner):
        if self.empty_after_train_epoch:
            torch.cuda.empty_cache()