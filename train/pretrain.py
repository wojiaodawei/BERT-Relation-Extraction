import logging
import os
import time
from itertools import combinations

import numpy as np
import torch
from matplotlib import pyplot as plt
from ml_utils.common import valncreate_dir
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from dataloaders.mtb_data_loader import MTBPretrainDataLoader
from model.albert.albert import AlbertModel
from model.bert.bert import BertModel
from src.train_funcs import Two_Headed_Loss

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
        self.train_len = len(self.data_loader.train_generator)
        logger.info("Loaded %d pre-training samples." % self.train_len)

        if "albert" in self.transformer:
            self.model = AlbertModel.from_pretrained(
                model_size=self.transformer,
                force_download=False,
                pretrained_model_name_or_path=self.transformer,
            )
        else:
            self.model = BertModel.from_pretrained(
                model_size=self.transformer,
                pretrained_model_name_or_path=self.transformer,
                force_download=False,
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
            normalize=True,
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
        self._best_mtb_bce = 1
        self._train_loss = []
        self._train_lm_acc = []
        self._lm_acc = []
        self._mtb_bce = []

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
                self._best_mtb_bce,
                self._train_loss,
                self._train_lm_acc,
            ) = self.load_best_model(checkpoint_dir)

        logger.info("Starting training process")
        pad_id = self.tokenizer.pad_token_id
        update_size = len(self.data_loader.train_generator) // 10
        for epoch in range(self._start_epoch, self.config.get("epochs")):
            start_time = time.time()
            self.model.train()
            train_loss = 0.0
            train_loss_per_batch = []
            train_acc = 0.0
            train_lm_acc_per_batch = []
            train_mtb_bce = 0.0
            train_mtb_bce_per_batch = []
            for i, data in enumerate(self.data_loader.train_generator):
                sequence, masked_label, e1_e2_start, blank_labels = data
                masked_label = masked_label[(masked_label != pad_id)]
                if masked_label.shape[0] == 0:
                    continue
                if self.train_on_gpu:
                    masked_label = masked_label.cuda()
                blanks_logits, lm_logits = self._get_logits(
                    e1_e2_start, sequence
                )

                if lm_logits.size(0) != masked_label.size(0):
                    blanks_logits, lm_logits = self._get_logits(
                        e1_e2_start, sequence
                    )

                loss = self.criterion(
                    lm_logits, blanks_logits, masked_label, blank_labels,
                )
                loss = loss / self.gradient_acc_steps

                loss.backward()

                clip_grad_norm_(
                    self.model.parameters(), self.config.get("max_norm")
                )

                if (i % self.gradient_acc_steps) == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss += loss.item()
                train_metrics = self.calculate_metrics(
                    lm_logits, blanks_logits, masked_label, blank_labels,
                )
                train_acc += train_metrics[0]
                train_mtb_bce += train_metrics[1]
                if (i % update_size) == (update_size - 1):
                    train_loss_per_batch.append(
                        self.gradient_acc_steps * train_loss / update_size
                    )
                    train_lm_acc_per_batch.append(train_acc / update_size)
                    train_mtb_bce_per_batch.append(train_mtb_bce / update_size)
                    logger.info(
                        "Epoch {0} - {1}/{2} points]:".format(
                            epoch + 1, (i + 1), self.train_len
                        )
                    )
                    logger.info(
                        "Train loss: {0}, Train LM accuracy: {1}, Train MTB Binary Cross Entropy {2}".format(
                            train_loss_per_batch[-1],
                            train_lm_acc_per_batch[-1],
                            train_mtb_bce_per_batch[-1],
                        )
                    )
                    train_loss = 0.0
                    train_acc = 0.0
                    train_mtb_bce = 0.0

            eval_result = self.evaluate()
            self._lm_acc += [eval_result[0]]
            self._mtb_bce += [eval_result[1]]

            self.scheduler.step()
            self._train_loss.append(np.mean(train_loss_per_batch))
            self._train_lm_acc.append(np.mean(train_lm_acc_per_batch))
            logger.info(
                "Epoch finished, took {0} seconds.".format(
                    time.time() - start_time
                )
            )
            logger.info("Loss at Epoch {0}".format(epoch + 1))
            logger.info("Loss: {0}".format(self._train_loss[-1]))
            logger.info(
                "Train LM Accuracy: {0}".format(self._train_lm_acc[-1])
            )
            logger.info("Validation LM Accuracy: {0}".format(self._lm_acc[-1]))
            logger.info(
                "Validation MTB Binary Cross Entropy: {0}".format(
                    self._mtb_bce[-1]
                )
            )

            if self._mtb_bce[-1] < self._best_mtb_bce:
                self._best_mtb_bce = self._mtb_bce[-1]
                self._save_model(checkpoint_dir, epoch, best_model=True)

            self._save_model(
                checkpoint_dir, epoch,
            )

        logger.info("Finished Training!")
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.scatter(
            np.arange(len(self._train_loss)), self._train_loss,
        )
        ax.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax.set_xlabel("Epoch", fontsize=22)
        ax.set_ylabel("Training Loss per batch", fontsize=22)
        ax.set_title("Training Loss vs Epoch", fontsize=32)
        plt.savefig(os.path.join("./data/", "loss_vs_epoch_ALBERT.png"))

        fig2 = plt.figure(figsize=(20, 20))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(
            np.arange(len(self._train_lm_acc)), self._train_lm_acc,
        )
        ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax2.set_xlabel("Epoch", fontsize=22)
        ax2.set_ylabel("Test Masked LM Accuracy", fontsize=22)
        ax2.set_title("Test Masked LM Accuracy vs Epoch", fontsize=32)
        plt.savefig(os.path.join("./data/", "accuracy_vs_epoch_ALBERT.png"))

        return self.model

    def _save_model(self, path, epoch, best_model: bool = False):
        if best_model:
            model_path = os.path.join(path, "best_model.pth.tar")
        else:
            model_path = os.path.join(
                path, "checkpoint_epoch_{0}.pth.tar"
            ).format(epoch + 1)
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "best_acc": self._train_lm_acc[-1],
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "losses_per_epoch": self._train_loss,
                "accuracy_per_epoch": self._train_loss,
                "lm_acc": self._lm_acc,
                "blanks_mse": self._mtb_bce,
            },
            model_path,
        )

    def _get_logits(self, e1_e2_start, x):
        attention_mask = (x != self.tokenizer.pad_token_id).float()
        token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
        if self.train_on_gpu:
            x = x.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
        blanks_logits, lm_logits = self.model(
            x,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            e1_e2_start=e1_e2_start,
        )
        lm_logits = lm_logits[(x == self.tokenizer.mask_token_id)]
        return blanks_logits, lm_logits

    def evaluate(self) -> tuple:
        """
        Run the validation generator and return performance metrics.
        """
        pad_id = self.tokenizer.pad_token_id
        total_loss = []
        lm_acc = []
        blanks_mse = []

        self.model.eval()
        with torch.no_grad():
            for data in self.data_loader.validation_generator:
                (x, masked_label, e1_e2_start, blank_labels) = data
                masked_label = masked_label[(masked_label != pad_id)]
                if masked_label.shape[0] == 0:
                    continue
                if self.train_on_gpu:
                    masked_label = masked_label.cuda()
                blanks_logits, lm_logits = self._get_logits(e1_e2_start, x)

                loss = self.criterion(
                    lm_logits, blanks_logits, masked_label, blank_labels,
                )

                total_loss += loss.cpu().numpy()
                eval_result = self.calculate_metrics(
                    lm_logits, blanks_logits, masked_label, blank_labels
                )
                lm_acc += [eval_result[0]]
                blanks_mse += [eval_result[1]]
        self.model.train()
        return (
            np.mean(lm_acc),
            sum(b for b in blanks_mse if b != 1)
            / len([b for b in blanks_mse if b != 1]),
        )

    def calculate_metrics(
        self, lm_logits, blanks_logits, masked_for_pred, blank_labels,
    ) -> tuple:
        """
        Calculates the performance metrics of the MTB model.

        Args:
            lm_logits: Language model Logits per word in vocabulary
            blanks_logits: Blank logits
            masked_for_pred: List of marked tokens
            blank_labels: Blank labels
        """
        lm_logits_pred_ids = torch.softmax(lm_logits, dim=-1).max(1)[1]
        lm_accuracy = (
            (lm_logits_pred_ids == masked_for_pred).sum().float()
            / len(masked_for_pred)
        ).item()

        pos_idxs = [
            i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 1
        ]
        neg_idxs = [
            i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 0
        ]

        if len(pos_idxs) > 1:
            # positives
            pos_logits = []
            for pos1, pos2 in combinations(pos_idxs, 2):
                pos_logits.append(
                    self._get_mtb_logits(
                        blanks_logits[pos1, :], blanks_logits[pos2, :]
                    )
                )
            pos_logits = torch.stack(pos_logits, dim=0)
            pos_labels = [1.0 for _ in range(pos_logits.shape[0])]
        else:
            pos_logits, pos_labels = torch.FloatTensor([]), []
            if blanks_logits.is_cuda:
                pos_logits = pos_logits.cuda()

        # negatives
        neg_logits = []
        for pos_idx in pos_idxs:
            for neg_idx in neg_idxs:
                neg_logits.append(
                    Pretrainer._get_mtb_logits(
                        blanks_logits[pos_idx, :], blanks_logits[neg_idx, :]
                    )
                )
        neg_logits = torch.stack(neg_logits, dim=0)
        neg_labels = [0.0 for _ in range(neg_logits.shape[0])]

        blank_labels_ = torch.FloatTensor(pos_labels + neg_labels)
        blank_pred = torch.cat([pos_logits, neg_logits], dim=0)
        bce = nn.BCEWithLogitsLoss(reduction="mean")(
            blank_pred.detach().cpu(), blank_labels_.detach().cpu()
        )

        return lm_accuracy, bce.numpy()

    @classmethod
    def _get_mtb_logits(cls, f1_vec, f2_vec):
        factor = 1 / (torch.norm(f1_vec) * torch.norm(f2_vec))
        return factor * torch.dot(f1_vec, f2_vec)
