import os 
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0, os.getcwd())

import torch

from src.utils import get_cfg_by_file
from src.eval import get_eval_dataframe_from_model
from src.dataset import CatDogDataset
from src.logger_helper import setup_logger
from src.transform import TestPreprocessor
from src.model import MODEL


logger = setup_logger(level=logging.INFO)

def main():
    image_path_list = [str(ele) for ele in Path(args.img_dir).glob("*.jpg")]
    image_path_list.sort(key=lambda s: int(os.path.basename(s).split(".")[0]))
    label_list = [0] * len(image_path_list)

    logger.info(f"Found {len(image_path_list)} images.")
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
        model.load_state_dict(torch.load(args.weight))
        logger.info(f"Load model weights from {args.weight} successfully.")

    df = get_eval_dataframe_from_model(infer_loader, model, loss_fn=None, device=config.device)
    df["id"] = df["name"].apply(lambda s: s.split(".")[0])
    df.rename(columns={"pred": "label"}, inplace=True)
    df[["id", "label"]].to_csv(args.save_path, index=False)
    logger.info(f"Save prediction result at {args.save_path}.") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference from image directory.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default="../dogs-vs-cats/train")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    config = get_cfg_by_file(args.config_file)
    main() 