import logging
import os
import time

import torch
from utils.constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL,
)
logger = logging.getLogger(__file__)


class RelationExtractor:
    def __init__(self):
        """
        Base Class for Relation extraction models using Transformers.
        """

    def load_best_model(self, checkpoint_dir: str):
        """
        Loads the current best model in the checkpoint directory.

        Args:
            checkpoint_dir: Checkpoint directory path
        """
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth.tar")
        logger.info("Loading best model from {0}".format(best_model_path))
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint

    def _prepare_epoch(self, epoch):
        logger.info("Starting epoch {0}".format(epoch + 1))
        self.model.train()
        return time.time()

    @classmethod
    def _reset_train_metrics(cls):
        train_loss = 0.0
        train_acc = 0.0
        return train_acc, train_loss

    def save_on_epoch_end(self, benchmark: list, baseline: int, epoch: int):
        """
        Saves current moddel at the end of the epoch.

        Writes also the best model if it's better than the current baseline

        Args:
            benchmark: List of benchmark results
            baseline: Current baseline. Best model performance so far
            epoch: Current epoch
        """
        self._save_model(
            self.checkpoint_dir, epoch,
        )
        if benchmark[-1] > baseline:
            self._save_model(self.checkpoint_dir, epoch, best_model=True)
            return benchmark[-1]

    def on_epoch_end(self, epoch: int, benchmark: list, baseline: int):
        """
        Function to run at the end of an epoch.

        Runs the evaluation method. Increments and the scheduler

        Args:
            epoch: Current epoch
            benchmark: List of benchmark results
            baseline: Current baseline. Best model performance so far
        """
        self.scheduler.step()
        eval_results = self.evaluate()
        new_baseline = self.save_on_epoch_end(benchmark, baseline, epoch)

        return eval_results, new_baseline
