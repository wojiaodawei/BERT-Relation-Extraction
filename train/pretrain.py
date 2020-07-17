import logging
import os
import time

import numpy as np
import torch
from ml_utils.common import valncreate_dir
from torch import optim as optim
from torch.nn.utils import clip_grad_norm_

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from dataloaders.mtb_data_loader import MTBPretrainDataLoader
from matplotlib import pyplot as plt
from src.model.ALBERT.modeling_albert import AlbertModel
from src.train_funcs import Two_Headed_Loss, evaluate_

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL,
)
logger = logging.getLogger(__file__)


class Pretrainer:
    def __init__(self, config: dict):
        """
        Matching the Blanks pretrainer.

        Args:
            config: configuration parameters
        """
        self.gradient_acc_steps = config.get("gradient_acc_steps")
        self.transformer = config.get("transformer")
        self.config = config
        self.data_loader = MTBPretrainDataLoader(self.config)
        self.data_loader.load_dataset()
        self.train_len = len(self.data_loader.train_generator)
        logger.info("Loaded %d pre-training samples." % self.train_len)

        self.model = AlbertModel.from_pretrained(
            model_size=self.transformer,
            force_download=False,
            pretrained_model_name_or_path=self.transformer,
        )

        self.tokenizer = self.data_loader.tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        if e1_id == e2_id == 1:
            raise ValueError("e1_id == e2_id == 1")

        self.train_on_gpu = torch.cuda.is_available()
        if self.train_on_gpu:
            self.model.cuda()

        self.criterion = Two_Headed_Loss(
            lm_ignore_idx=self.tokenizer.pad_token_id,
            use_logits=True,
            normalize=False,
        )
        self.optimizer = optim.Adam(
            [{"params": self.model.parameters(), "lr": self.config.get("lr")}]
        )

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30],
            gamma=0.8,
        )

        self._start_epoch = 0
        self._best_pred = 0
        self._losses_per_epoch = []
        self._accuracy_per_epoch = []

    def load_best_model(self, checkpoint_dir: str):
        """
        Loads the current best model in the checkpoint directory.

        Args:
            checkpoint_dir: Checkpoint directory path
        """
        logger.info(
            "Loading best model from {0}".format(
                os.path.join(checkpoint_dir, "best_model.pth.tar")
            )
        )
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, "best_model.pth.tar")
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return (
            checkpoint["epoch"],
            checkpoint["best_acc"],
            checkpoint["losses_per_epoch"],
            checkpoint["accuracy_per_epoch"],
        )

    def train(self):
        """
        Runs the training.
        """
        checkpoint_dir = os.path.join(
            "models", "pretraining", self.transformer
        )
        valncreate_dir(checkpoint_dir)
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth.tar")
        resume = self.config.get("resume", False)
        if resume and os.path.exists(best_model_path):
            (
                self._start_epoch,
                self._best_pred,
                self._losses_per_epoch,
                self._accuracy_per_epoch,
            ) = self.load_best_model(checkpoint_dir)

        logger.info("Starting training process")
        pad_id = self.tokenizer.pad_token_id
        mask_id = self.tokenizer.mask_token_id
        update_size = len(self.data_loader.train_generator) // 10
        for epoch in range(self._start_epoch, self.config.get("epochs")):
            start_time = time.time()
            self.model.train()
            total_loss = 0.0
            losses_per_batch = []
            total_acc = 0.0
            lm_accuracy_per_batch = []
            for i, data in enumerate(self.data_loader.train_generator):
                (x, masked_for_pred, e1_e2_start, blank_labels) = data
                masked_for_pred = masked_for_pred[(masked_for_pred != pad_id)]
                if masked_for_pred.shape[0] == 0:
                    logger.warning("Empty dataset. Skipping")
                    continue
                attention_mask = (x != pad_id).float()
                token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

                if self.train_on_gpu:
                    x = x.cuda()
                    masked_for_pred = masked_for_pred.cuda()
                    attention_mask = attention_mask.cuda()
                    token_type_ids = token_type_ids.cuda()

                blanks_logits, lm_logits = self.model(
                    x,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    Q=None,
                    e1_e2_start=e1_e2_start,
                )
                lm_logits = lm_logits[(x == mask_id)]

                if (i % update_size) == (update_size - 1):
                    verbose = True
                else:
                    verbose = False

                loss = self.criterion(
                    lm_logits,
                    blanks_logits,
                    masked_for_pred,
                    blank_labels,
                    verbose=verbose,
                )
                loss = loss / self.gradient_acc_steps

                loss.backward()

                clip_grad_norm_(
                    self.model.parameters(), self.config.get("max_norm")
                )

                if (i % self.gradient_acc_steps) == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()
                total_acc += evaluate_(
                    lm_logits,
                    blanks_logits,
                    masked_for_pred,
                    blank_labels,
                    self.tokenizer,
                    print_=False,
                )[0]

                if (i % update_size) == (update_size - 1):
                    losses_per_batch.append(
                        self.gradient_acc_steps * total_loss / update_size
                    )
                    lm_accuracy_per_batch.append(total_acc / update_size)
                    print(
                        "[Epoch: %d, %5d/ %d points] total loss, lm accuracy per batch: %.3f, %.3f"
                        % (
                            epoch + 1,
                            (i + 1),
                            self.train_len,
                            losses_per_batch[-1],
                            lm_accuracy_per_batch[-1],
                        )
                    )
                    total_loss = 0.0
                    total_acc = 0.0
                    logger.info(
                        "Last batch samples (pos, neg): %d, %d"
                        % (
                            (blank_labels.squeeze() == 1).sum().item(),
                            (blank_labels.squeeze() == 0).sum().item(),
                        )
                    )

            self.scheduler.step()
            self._losses_per_epoch.append(np.mean(losses_per_batch))
            self._accuracy_per_epoch.append(np.mean(lm_accuracy_per_batch))
            print(
                "Epoch finished, took %.2f seconds."
                % (time.time() - start_time)
            )
            print(
                "Losses at Epoch %d: %.7f"
                % (epoch + 1, self._losses_per_epoch[-1])
            )
            print(
                "Accuracy at Epoch %d: %.7f"
                % (epoch + 1, self._accuracy_per_epoch[-1])
            )

            if self._accuracy_per_epoch[-1] > self._best_pred:
                self._best_pred = self._accuracy_per_epoch[-1]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "best_acc": self._accuracy_per_epoch[-1],
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "losses_per_epoch": self._losses_per_epoch,
                        "accuracy_per_epoch": self._losses_per_epoch,
                    },
                    os.path.join(checkpoint_dir, "best_model.pth.tar"),
                )

            checkpoint_dir = os.path.join(
                "models", "pretraining", self.transformer
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_acc": self._accuracy_per_epoch[-1],
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "losses_per_epoch": self._losses_per_epoch,
                    "accuracy_per_epoch": self._losses_per_epoch,
                },
                os.path.join(
                    checkpoint_dir, "checkpoint_epoch_{0}.pth.tar"
                ).format(epoch + 1),
            )

        logger.info("Finished Training!")
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.scatter(
            np.arange(len(self._losses_per_epoch)), self._losses_per_epoch,
        )
        ax.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax.set_xlabel("Epoch", fontsize=22)
        ax.set_ylabel("Training Loss per batch", fontsize=22)
        ax.set_title("Training Loss vs Epoch", fontsize=32)
        plt.savefig(os.path.join("./data/", "loss_vs_epoch_ALBERT.png"))

        fig2 = plt.figure(figsize=(20, 20))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(
            np.arange(len(self._accuracy_per_epoch)), self._accuracy_per_epoch,
        )
        ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax2.set_xlabel("Epoch", fontsize=22)
        ax2.set_ylabel("Test Masked LM Accuracy", fontsize=22)
        ax2.set_title("Test Masked LM Accuracy vs Epoch", fontsize=32)
        plt.savefig(os.path.join("./data/", "accuracy_vs_epoch_ALBERT.png"))

        return self.model
