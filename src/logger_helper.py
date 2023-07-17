import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path

LOGDIR = "./log"

log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - %(pathname)s:%(lineno)d : %(message)s"
log_console_format = "[%(levelname)s] - %(asctime)s - %(name)s : %(message)s"


def setup_logger(folder_name: str=LOGDIR, level=logging.DEBUG):
    file_name = _set_folder_and_get_logging_name(folder_name)
    logger = logging.getLogger()
    logger.setLevel(level)
    console_formatter = logging.Formatter(log_console_format, datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(console_formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    if file_name:
        fh = logging.FileHandler(file_name)
        file_formater = logging.Formatter(log_file_format, datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(file_formater)
        logger.addHandler(fh)
    return logger


def _set_folder_and_get_logging_name(folder_name: str) -> Path:
    path = Path(folder_name)
    if not path.is_dir():
        os.mkdir(path)
    filename = path / (datetime.now().strftime("%Y%m%dT%H%M%S") + ".log")
    return filename