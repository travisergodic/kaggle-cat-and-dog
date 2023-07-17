import timm.scheduler
import torch.optim.lr_scheduler as lr_scheduler

from src.registry import SCHEDULER


@SCHEDULER.register('StepLR')
def build_StepLR(optimizer, **kwargs):
    return lr_scheduler.StepLR(optimizer, **kwargs)

@SCHEDULER.register('LambdaLR')
def build_LambdaLRScheduler(optimizer, **kwargs):
    return lr_scheduler.LambdaLR(optimizer, **kwargs)

@SCHEDULER.register('CosineLR')
def build_CosineLRScheduler(optimizer, **kwargs):
    return timm.scheduler.CosineLRScheduler(optimizer, **kwargs)