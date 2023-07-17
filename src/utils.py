import os
import sys
import importlib
import logging

import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_cfg_by_file(cfg_file):
    try:
        sys.path.append(os.path.dirname(cfg_file))
        current_cfg = importlib.import_module(os.path.basename(cfg_file).split(".")[0])
        logger.info(f'Import {cfg_file} successfully!')
    except Exception:
        raise ImportError(f'Fail to import {cfg_file}')
    return current_cfg


def plot_loss_history_from_trainer(trainer, save_path):
    plt.plot(range(1, trainer.n_epochs+1), trainer.epoch_train_records['loss'], color='blue', label='train loss')
    plt.plot(range(1, trainer.n_epochs+1), trainer.epoch_test_records['loss'], color='orange', label='val loss')
    plt.legend()
    plt.savefig(save_path)
    plt.clf()


def save_performance_history_dataframe_from_trainer(trainer, save_path):
    df = pd.DataFrame(dict(trainer.epoch_test_records))
    df.index = range(1, trainer.n_epochs + 1)
    df.to_csv(save_path)


def save_performance_dataframe_from_trainer(trainer, save_path):
    df = pd.DataFrame(dict(trainer.epoch_test_records))
    df.index = range(1, trainer.n_epochs + 1)
    df = df.loc[[trainer.n_epochs, trainer.best_epoch], :].copy()
    df.index = ['last', 'best'] 
    df.to_csv(save_path)


def plot_roc_curve(y, prob, save_path):
    fpr, tpr, threshold = metrics.roc_curve(y, prob)
    roc_auc = metrics.auc(fpr, tpr)
    import matplotlib.pyplot as plt
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1],"r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(save_path)
    plt.clf()


def plot_confusion_table(y, pred, save_path):
    conf = metrics.confusion_matrix(y, pred) 
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.set(font_scale=1.4)
    conf_plot = sns.heatmap(conf, cmap=colormap, annot=True, cbar=False, fmt="d")
    conf_fig = conf_plot.get_figure()
    plt.title("confusion matrix") 
    conf_fig.savefig(save_path)
    plt.clf()


def image_path_to_class_id(path):
    name = os.path.basename(path)
    return {"cat": 0, "dog": 1}[name.split(".")[0]]