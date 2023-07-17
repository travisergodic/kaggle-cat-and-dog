import os
import logging
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from src.eval import get_performance_dict

# logger
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self, model, optimizer, iter_hook, loss_fn, device, n_epochs, iter_scheduler=None, 
            epoch_scheduler=None, save_freq=5, checkpoint_dir=None, monitor='loss'
        ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.iter_scheduler = iter_scheduler
        self.epoch_scheduler = epoch_scheduler
        self.iter_hook = iter_hook
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.best_epoch = 0
        self.epoch_train_records = defaultdict(list)
        self.epoch_test_records = defaultdict(list)

    def fit(self, train_loader, test_loader=None):
        for self.epoch in range(1, self.n_epochs + 1): 
            self.model.train()
            pbar = tqdm(train_loader)
            pbar.set_description(f"Epoch {self.epoch}/{self.n_epochs}") 
            iter_train_loss_records = []
            
            for self.iter, batch in enumerate(pbar):
                X, y = batch["data"].to(self.device), batch["label"].to(self.device)
                train_loss = self.iter_hook.run_iter(self, X, y).item()
                pbar.set_postfix(loss=train_loss)
                iter_train_loss_records.append(train_loss)
                if self.iter_scheduler is not None:
                    self.iter_scheduler.step(self.epoch + self.iter/len(train_loader))

            self.epoch_train_records['loss'].append(np.mean(iter_train_loss_records))
            logger.info(f'train_loss: {self.epoch_train_records["loss"][-1]}')  

            # update scheduler
            if self.epoch_scheduler is not None:
                self.epoch_scheduler.step()

            # evaluate
            if test_loader is not None:
                self.test(test_loader)
                
            # save best checkpoint
            if test_loader is not None:
                criterion = min if self.monitor == 'loss' else max
                if self.epoch_test_records[self.monitor][-1] == criterion(self.epoch_test_records[self.monitor]):
                    self.best_epoch = self.epoch
                    Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best.pt'))
            
            # save last checkpoint
            if self.epoch % self.save_freq == 0:
                Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'last.pt'))

    @torch.no_grad()
    def test(self, test_loader):        
        performance_dict = get_performance_dict(test_loader, self.model, self.loss_fn, threshold=0.5, device=self.device)
        for metric_name, performance in performance_dict.items():
            self.epoch_test_records[metric_name].append(performance)
        logger.info(f'performance: {performance_dict}')
