import logging

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

logger = logging.getLogger(__name__)


def get_auc_score(preds, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    return metrics.auc(fpr, tpr)

def get_confusion_matrix(y, pred):
    return metrics.confusion_matrix(y, pred)

def get_auc_score(preds, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    return metrics.auc(fpr, tpr)

@torch.no_grad()
def get_eval_dataframe_from_model(loader, model, loss_fn=None, device="cpu"):
    prob_list, target_list, loss_list, name_list = [], [], [], [] 
    # model
    model.to(device)
    model.eval()

    # create pair sample loader
    for batch in tqdm(loader):
        X, y = batch["data"].to(device), batch["label"].to(device)
        names = batch["name"]
        pred = model(X)

        if loss_fn: 
            loss_list.extend([loss_fn(pred, y).item()] * len(names))
        prob_list.append(pred)
        target_list.append(y)
        name_list.extend(names)

    prob_arr = torch.cat(prob_list, dim=0).cpu().numpy()
    target_arr = torch.cat(target_list, dim=0).cpu().numpy()
    name_arr = np.array(name_list)
    pred_arr = prob_arr.argmax(axis=1)
    data_dict = {"name": name_arr, "pred": pred_arr, "gt": target_arr, "prob": prob_arr[:, 1]}
    if loss_fn:
        data_dict["loss"] = np.array(loss_list)
    return pd.DataFrame(data_dict)

def get_performance_dict(loader, model, loss_fn=None, threshold=0.5, device='cpu'):
    df = get_eval_dataframe_from_model(loader, model, loss_fn, device)
    preds = np.where(df['prob'].values < threshold, 0, 1)
    res = {
        'auc': get_auc_score(df['prob'].values, df['gt'].values),
        'accuracy': metrics.accuracy_score(df['gt'].values, preds), 
        'recall': metrics.recall_score(df['gt'].values, preds), 
        'precision': metrics.precision_score(df['gt'].values, preds) 
    }
    if loss_fn:
        res['loss'] = np.mean(df['loss'].values)
    return res