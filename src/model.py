# swin: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py
# convnext: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py
# beit: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/beit.py

import timm 
import torchvision
import torch
import torch.nn as nn 

from src.registry import MODEL


def load_model(model, checkpoint_path):
    return model.load_state_dict(torch.load(checkpoint_path))


@MODEL.register("resnet50")
def build_resnet50(num_classes=1, pretrain=True, dropout=0, checkpoint_path=None):
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrain else None
    )
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout), nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    )
    if checkpoint_path is not None:
        load_model(model, checkpoint_path)
    return model


@MODEL.register("swin_transformer")
def build_swin_transformer(name, num_classes, pretrain=True, dropout=0, checkpoint_path=None):
    model = timm.create_model(name, pretrained=pretrain)
    model.head.drop = nn.Dropout(p=dropout)
    in_features = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    if checkpoint_path is not None:
        load_model(model, checkpoint_path)
    return model


@MODEL.register("convnext")
def build_convnext(name, num_classes=1, pretrain=True, dropout=0, checkpoint_path=None):
    model = timm.create_model(name, pretrained=pretrain)
    model.head.drop = nn.Dropout(p=dropout)
    in_features = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    if checkpoint_path is not None:
        load_model(model, checkpoint_path)
    return model


@MODEL.register("efficientnetv2")
def build_efficientnetv2(name, num_classes=1, pretrain=True, dropout=0, checkpoint_path=None):
    model = timm.create_model(name, pretrained=pretrain)
    model.bn2.drop = nn.Dropout(p=dropout)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    if checkpoint_path is not None:
        load_model(model, checkpoint_path)
    return model


@MODEL.register("beitv2")
def build_beit(name, num_classes=1, pretrain=True, dropout=0, checkpoint_path=None):
    model = timm.create_model(name, pretrained=pretrain)
    model.head_drop = nn.Dropout(p=dropout)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    if checkpoint_path is not None:
        load_model(model, checkpoint_path)
    return model