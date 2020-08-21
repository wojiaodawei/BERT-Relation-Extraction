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


class MTBModel:
    def __init__(self, config: dict):
        """
        Matching the Blanks Model.

        Args:
            config: configuration parameters
        """
        self.experiment_name = config.get("experiment_name")
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
        self.checkpoint_dir = os.path.join(
            "models", "pretraining", self.experiment_name, self.transformer
        )
        valncreate_dir(self.checkpoint_dir)

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
        return (
            checkpoint["epoch"],
            checkpoint["best_mtb_bce"],
            checkpoint["losses_per_epoch"],
            checkpoint["accuracy_per_epoch"],
            checkpoint["lm_acc"],
            checkpoint["blanks_mse"],
        )

    def train(self):
        """
        Runs the training.
        """
        best_model_path = os.path.join(
            self.checkpoint_dir, "best_model.pth.tar"
        )
        resume = self.config.get("resume", False)
        if resume and os.path.exists(best_model_path):
            (
                self._start_epoch,
                self._best_mtb_bce,
                self._train_loss,
                self._train_lm_acc,
                self._lm_acc,
                self._mtb_bce,
            ) = self.load_best_model(self.checkpoint_dir)

        logger.info("Starting training process")
        update_size = len(self.data_loader.train_generator) // 10
        for epoch in range(self._start_epoch, self.config.get("epochs")):
            self._train_epoch(epoch, update_size)
            self._plot_results()
        logger.info("Finished Training.")
        return self.model

    def _plot_results(self):
        results_path = os.path.join(
            "results", "pretraining", self.experiment_name, self.transformer
        )
        valncreate_dir(results_path)
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.scatter(
            np.arange(len(self._train_loss)), self._train_loss,
        )
        ax.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax.set_xlabel("Epoch", fontsize=22)
        ax.set_ylabel("Training Loss per batch", fontsize=22)
        ax.set_title("Training Loss vs Epoch", fontsize=32)
        plt.savefig(
            os.path.join(
                results_path, "train_loss_{0}.png".format(self.transformer)
            )
        )

        fig2 = plt.figure(figsize=(20, 20))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(
            np.arange(len(self._train_lm_acc)), self._train_lm_acc,
        )
        ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax2.set_xlabel("Epoch", fontsize=22)
        ax2.set_ylabel("Train Masked LM Accuracy", fontsize=22)
        ax2.set_title("Train Masked LM Accuracy vs Epoch", fontsize=32)
        plt.savefig(
            os.path.join(
                results_path, "train_lm_acc_{0}.png".format(self.transformer)
            )
        )

        fig3 = plt.figure(figsize=(20, 20))
        ax3 = fig3.add_subplot(111)
        ax3.scatter(
            np.arange(len(self._lm_acc)), self._lm_acc,
        )
        ax3.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax3.set_xlabel("Epoch", fontsize=22)
        ax3.set_ylabel("Val Masked LM Accuracy", fontsize=22)
        ax3.set_title("Val Masked LM Accuracy vs Epoch", fontsize=32)
        plt.savefig(
            os.path.join(
                results_path, "val_lm_acc_{0}.png".format(self.transformer)
            )
        )

        fig2 = plt.figure(figsize=(20, 20))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(
            np.arange(len(self._mtb_bce)), self._mtb_bce,
        )
        ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax2.set_xlabel("Epoch", fontsize=22)
        ax2.set_ylabel("Val MTB Binary Cross Entropy", fontsize=22)
        ax2.set_title("Val MTB Binary Cross Entropy", fontsize=32)
        plt.savefig(
            os.path.join(
                results_path, "val_mtb_bce_{0}.png".format(self.transformer)
            )
        )

    def _train_epoch(self, epoch, update_size):
        logger.info("Starting epoch {0}".format(epoch + 1))
        start_time = time.time()

        self.model.train()

        train_acc, train_loss, train_mtb_bce = MTBModel._reset_train_metrics()
        train_loss_per_batch = []
        train_lm_acc_per_batch = []
        train_mtb_bce_per_batch = []

        for i, data in enumerate(self.data_loader.train_generator):
            sequence, masked_label, e1_e2_start, blank_labels = data
            do_updates = (i % self.gradient_acc_steps) == 0
            res = self._train_on_batch(
                sequence, masked_label, e1_e2_start, blank_labels, do_updates
            )
            if res[0]:
                train_loss += res[0]
                train_acc += res[1]
                train_mtb_bce += res[2]
            if (i % update_size) == (update_size - 1):
                train_loss_per_batch.append(
                    self.gradient_acc_steps * train_loss / update_size
                )
                train_lm_acc_per_batch.append(train_acc / update_size)
                train_mtb_bce_per_batch.append(train_mtb_bce / update_size)
                logger.info(
                    "{0}/{1} pools: - ".format((i + 1), self.train_len)
                    + "Train loss: {0}, Train LM accuracy: {1}, Train MTB Binary Cross Entropy {2}".format(
                        train_loss_per_batch[-1],
                        train_lm_acc_per_batch[-1],
                        train_mtb_bce_per_batch[-1],
                    )
                )
                (
                    train_acc,
                    train_loss,
                    train_mtb_bce,
                ) = MTBModel._reset_train_metrics()
        eval_result = self.evaluate()
        self._lm_acc += [eval_result[0]]
        self._mtb_bce += [eval_result[1]]
        self.scheduler.step()
        self._train_loss.append(np.mean(train_loss_per_batch))
        self._train_lm_acc.append(np.mean(train_lm_acc_per_batch))
        logger.info(
            "Epoch {0} finished, took {1} seconds.".format(
                epoch, time.time() - start_time
            )
        )
        logger.info("Loss: {0}".format(self._train_loss[-1]))
        logger.info("Train LM Accuracy: {0}".format(self._train_lm_acc[-1]))
        logger.info("Validation LM Accuracy: {0}".format(self._lm_acc[-1]))
        logger.info(
            "Validation MTB Binary Cross Entropy: {0}".format(
                self._mtb_bce[-1]
            )
        )
        if self._mtb_bce[-1] < self._best_mtb_bce:
            self._best_mtb_bce = self._mtb_bce[-1]
            self._save_model(self.checkpoint_dir, epoch, best_model=True)
        self._save_model(
            self.checkpoint_dir, epoch,
        )

    @classmethod
    def _reset_train_metrics(cls):
        train_loss = 0.0
        train_acc = 0.0
        train_mtb_bce = 0.0
        return train_acc, train_loss, train_mtb_bce

    def _train_on_batch(
        self,
        sequence,
        masked_label,
        e1_e2_start,
        blank_labels,
        do_update: bool = True,
    ):
        masked_label = masked_label[
            (masked_label != self.tokenizer.pad_token_id)
        ]
        if masked_label.shape[0] == 0:
            return None, None, None
        if self.train_on_gpu:
            masked_label = masked_label.cuda()
        blanks_logits, lm_logits = self._get_logits(e1_e2_start, sequence)
        loss = self.criterion(
            lm_logits, blanks_logits, masked_label, blank_labels,
        )
        loss = loss / self.gradient_acc_steps
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.config.get("max_norm"))
        if do_update:
            self.optimizer.step()
            self.optimizer.zero_grad()
        train_metrics = self.calculate_metrics(
            lm_logits, blanks_logits, masked_label, blank_labels,
        )
        return loss.item(), train_metrics[0], train_metrics[1]

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
                "best_mtb_bce": self._best_mtb_bce,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "losses_per_epoch": self._train_loss,
                "accuracy_per_epoch": self._train_lm_acc,
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
        total_loss = []
        lm_acc = []
        blanks_mse = []

        self.model.eval()
        with torch.no_grad():
            for data in self.data_loader.validation_generator:
                (x, masked_label, e1_e2_start, blank_labels) = data
                masked_label = masked_label[
                    (masked_label != self.tokenizer.pad_token_id)
                ]
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

        pos_idxs = np.where(blank_labels == 1)[0]
        neg_idxs = np.where(blank_labels == 0)[0]

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
                    MTBModel._get_mtb_logits(
                        blanks_logits[pos_idx, :], blanks_logits[neg_idx, :]
                    )
                )
        neg_logits = torch.stack(neg_logits, dim=0)
        neg_labels = [0.0 for _ in range(neg_logits.shape[0])]

        blank_labels = torch.FloatTensor(pos_labels + neg_labels)
        blank_pred = torch.cat([pos_logits, neg_logits], dim=0)
        bce = nn.BCEWithLogitsLoss(reduction="mean")(
            blank_pred.detach().cpu(), blank_labels.detach().cpu()
        )

        return lm_accuracy, bce.numpy()

    @classmethod
    def _get_mtb_logits(cls, f1_vec, f2_vec):
        factor = 1 / (torch.norm(f1_vec) * torch.norm(f2_vec))
        return factor * torch.dot(f1_vec, f2_vec)
