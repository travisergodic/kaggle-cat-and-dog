import torch.optim as optim

from src.registry import OPTIMIZER
from .sam import SAM


@OPTIMIZER.register('Adam')
def build_Adam(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return optim.Adam(params, **kwargs)

@OPTIMIZER.register('AdamW')
def build_AdamW(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return optim.AdamW(params, **kwargs)

@OPTIMIZER.register('SGD')
def build_SGD(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return optim.SGD(params, **kwargs)

@OPTIMIZER.register('SAM_Adam')
def build_SAM_Adam(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return SAM(params, optim.Adam, **kwargs)

@OPTIMIZER.register('SAM_AdamW')
def build_SAM_AdamW(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return SAM(params, optim.AdamW, **kwargs)

@OPTIMIZER.register('SAM_SGD')
def build_SAM_SGD(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return SAM(params, optim.SGD, **kwargs)