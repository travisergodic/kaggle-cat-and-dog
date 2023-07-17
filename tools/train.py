import os
import logging
import argparse
from pathlib import Path
import sys
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset import CatDogDataset
from src.logger_helper import setup_logger
from src.utils import get_cfg_by_file, image_path_to_class_id, plot_loss_history_from_trainer, \
    save_performance_dataframe_from_trainer, save_performance_history_dataframe_from_trainer, \
    plot_roc_curve, plot_confusion_table
from src.eval import get_eval_dataframe_from_model
from src.trainer import Trainer
from src.iter_hook import *
from src.model import MODEL
from src.optimizer import OPTIMIZER
from src.scheduler import SCHEDULER
from src.transform import TrainPreprocessor, TestPreprocessor


logger = setup_logger(level=logging.INFO)


def main():
    image_path_list = np.array([str(ele) for ele in Path(args.img_dir).glob("*.jpg")])
    label_list = np.array([image_path_to_class_id(path) for path in image_path_list])

    if args.debug:
        indicies = np.random.choice(len(image_path_list), 500)
        image_path_list = image_path_list[indicies]
        label_list = label_list[indicies]

    # train test split
    train_path_list, test_path_list, train_label, test_label = train_test_split(
        image_path_list, label_list, test_size=args.test_ratio, 
        random_state=42, stratify=label_list
    )

    logger.info(f"Train: {len(train_path_list)} images.")
    logger.info(f"Test: {len(test_path_list)} images.")

    Path(f"./checkpoints/{args.exp_name}/").mkdir(exist_ok=True, parents=True)
    pd.DataFrame(
        {"name": [os.path.basename(path) for path in test_path_list], "label": test_label}
    ).to_csv(f"./checkpoints/{args.exp_name}/validation_set.csv", index=False)
    
    logger.info(f"Save validation_set.csv at ./checkpoints/{args.exp_name}/")

    train_dataset = CatDogDataset(
        train_path_list, train_label, num_classes=config.num_classes, image_transform=TestPreprocessor(img_size=config.img_size)
    )
    test_dataset = CatDogDataset(
        test_path_list, test_label, num_classes=config.num_classes, image_transform=TrainPreprocessor(img_size=config.img_size)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, 
        num_workers=config.num_workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs * 2, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True
    )

    # model
    model = MODEL.build(**config.model_cfg)

    # optimizer
    if args.adam:
        config.optimizer_cfg["type"] = "Adam"
    elif args.adamw:
        config.optimizer_cfg["type"] = "AdamW"

    if args.lr:
        config.optimizer_cfg["lr"] = args.lr

    if args.weight_decay:
        config.optimizer_cfg["weight_decay"] = args.weight_decay
    
    optimizer = OPTIMIZER.build(model=model, **config.optimizer_cfg)

    # load checkpoint
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
        logger.info(f"Load model weights from {args.weight} successfully.")

    # scheduler
    epoch_scheduler = SCHEDULER.build(optimizer=optimizer, **config.epoch_scheduler_cfg)
    iter_scheduler = SCHEDULER.build(optimizer=optimizer, **config.iter_scheduler_cfg)

    # iter hook
    iter_hook = SamIterHook() if type(optimizer).__name__.startswith("SAM") else NormalIterHook()
    logger.info(f"Use {type(iter_hook).__name__} object for each training iteration.")

    # build trainer
    trainer = Trainer(
        model=model, 
        optimizer=optimizer,
        iter_hook=iter_hook,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=config.device,
        n_epochs=config.n_epochs,
        iter_scheduler=iter_scheduler,
        epoch_scheduler=epoch_scheduler, 
        save_freq=config.save_freq,
        checkpoint_dir=f"./checkpoints/{args.exp_name}",
        monitor=config.monitor
    )

    trainer.fit(train_loader, test_loader)

    # save result
    plot_loss_history_from_trainer(trainer, f"./checkpoints/{args.exp_name}/loss_history.jpg")
    save_performance_dataframe_from_trainer(trainer, f"./checkpoints/{args.exp_name}/performance.csv")
    save_performance_history_dataframe_from_trainer(trainer, f"./checkpoints/{args.exp_name}/performance_history.csv")

    # plot roc and confusion matrix
    df = get_eval_dataframe_from_model(test_loader, model, None, config.device)
    y, pred, prob = df["gt"].values, df["pred"].values, df["prob"].values
    plot_roc_curve(y, prob, f"./checkpoints/{args.exp_name}/roc_curve.jpg")
    plot_confusion_table(y, pred, f"./checkpoints/{args.exp_name}/confusion_table.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default="../dogs-vs-cats/train")
    parser.add_argument("--test_ratio", type=str, default=0.15)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--adam", action="store_true")
    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = get_cfg_by_file(args.config_file)
    main() 