import logging
import os

import torch
from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

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
