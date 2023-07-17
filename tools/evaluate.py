import os 
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0, os.getcwd())

import torch
import pandas as pd
from sklearn import metrics

from src.utils import get_cfg_by_file, plot_confusion_table, plot_roc_curve
from src.eval import get_eval_dataframe_from_model, get_auc_score
from src.dataset import CatDogDataset
from src.logger_helper import setup_logger
from src.transform import TestPreprocessor
from src.model import MODEL


logger = setup_logger(level=logging.INFO)

def main():
    image_path_list, label_list = [], []
    unknown_path_list = []

    df = pd.read_csv(args.csv_path)

    for _, row in df.iterrows():
        name, label = row["name"], row["label"]
        image_path = os.path.join(args.img_dir, name)
        if os.path.isfile(image_path):
            image_path_list.append(image_path)
            label_list.append(int(label))
        else:
            unknown_path_list.append(image_path_list)

    logger.info(f"Found {len(image_path_list)} valid images.")
    logger.info(f"Found {len(unknown_path_list)} invalid images: \n {unknown_path_list}")

    infer_dataset = CatDogDataset(
        image_path_list, label_list, num_classes=config.num_classes, image_transform=TestPreprocessor(img_size=(224, 224))
    )

    infer_loader = torch.utils.data.DataLoader(
        infer_dataset, batch_size=args.bs, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True
    )

    # model
    model = MODEL.build(**config.model_cfg)

    # load checkpoint
    if args.weight is not None:
        res = model.load_state_dict(torch.load(args.weight))
        logger.info(f"Load model weights from {args.weight} successfully: {res}")

    # mkdir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # plot roc and confusion matrix
    df = get_eval_dataframe_from_model(infer_loader, model, loss_fn=None, device=config.device)
    performance_dict = {
        "auc": get_auc_score(df["prob"].values, df["gt"].values),
        "accuracy": metrics.accuracy_score(df["gt"].values, df["pred"].values), 
        'recall': metrics.recall_score(df["gt"].values, df["pred"].values), 
        'precision': metrics.precision_score(df['gt'].values, df["pred"].values) 
    }

    logger.info(f"Model performance: {performance_dict}")

    plot_roc_curve(df["gt"].values, df["prob"].values, f"{str(Path(args.save_dir) / 'roc_curve.jpg')}")
    logger.info(f"Save roc plot successfully.")

    plot_confusion_table(df["gt"].values, df["pred"].values, f"{str(Path(args.save_dir) / 'confusion_table.jpg')}")
    logger.info(f"Save confusion plot successfully.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference from image directory.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default="../dogs-vs-cats/train")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    config = get_cfg_by_file(args.config_file)
    main() 