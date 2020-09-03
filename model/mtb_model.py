import logging
import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from ml_utils.path_operations import valncreate_dir
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from dataloaders.mtb_data_loader import MTBPretrainDataLoader
from model.bert import BertModel
from model.relation_extractor import RelationExtractor
from src.train_funcs import Two_Headed_Loss

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATETIME_FORMAT,
    level=LOG_LEVEL,
)
logger = logging.getLogger(__file__)

sns.set(font_scale=2.2)


class MTBModel(RelationExtractor):
    def __init__(self, config: dict):
        """
        Matching the Blanks Model.

        Args:
            config: configuration parameters
        """
        super().__init__()
        self.experiment_name = config.get("experiment_name")
        self.transformer = config.get("transformer")
        self.config = config
        self.data_loader = MTBPretrainDataLoader(self.config)
        self.train_len = len(self.data_loader.train_generator)
        logger.info("Loaded %d pre-training samples." % self.train_len)

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

        self.train_on_gpu = torch.cuda.is_available() and config.get(
            "use_cuda", True
        )
        if self.train_on_gpu:
            logger.info("Train on GPU")
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

        self._points_seen = 0

    def load_best_model(self, checkpoint_dir: str):
        """
        Loads the current best model in the checkpoint directory.

        Args:
            checkpoint_dir: Checkpoint directory path
        """
        checkpoint = super().load_best_model(checkpoint_dir)
        return (
            checkpoint["epoch"],
            checkpoint["best_mtb_bce"],
            checkpoint["losses_per_epoch"],
            checkpoint["accuracy_per_epoch"],
            checkpoint["lm_acc"],
            checkpoint["blanks_mse"],
        )

    def train(self, **kwargs):
        """
        Runs the training.
        """
        save_best_model_only = kwargs.get("save_best_model_only", False)
        results_path = os.path.join(
            "results", "pretraining", self.experiment_name, self.transformer
        )
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
            self._train_epoch(epoch, update_size, save_best_model_only)
            data = self._write_kpis(results_path)
            self._plot_results(data, results_path)
        logger.info("Finished Training.")
        return self.model

    def _plot_results(self, data, save_at):
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.lineplot(x="Epoch", y="Train Loss", ax=ax, data=data, linewidth=4)
        ax.set_title("Training Loss")
        plt.savefig(
            os.path.join(
                save_at, "train_loss_{0}.png".format(self.transformer)
            )
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(20, 20))
        sns.lineplot(
            x="Epoch", y="Val MTB Loss", ax=ax, data=data, linewidth=4
        )
        ax.set_title("Val MTB Binary Cross Entropy")
        plt.savefig(
            os.path.join(
                save_at, "val_mtb_bce_{0}.png".format(self.transformer)
            )
        )
        plt.close(fig)

        tmp = data[["Epoch", "Train LM Accuracy", "Val LM Accuracy"]].melt(
            id_vars="Epoch", var_name="Set", value_name="LM Accuracy"
        )
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.lineplot(
            x="Epoch",
            y="LM Accuracy",
            hue="Set",
            ax=ax,
            data=tmp,
            linewidth=4,
        )
        ax.set_title("LM Accuracy")
        plt.savefig(
            os.path.join(save_at, "lm_acc_{0}.png".format(self.transformer))
        )
        plt.close(fig)

    def _write_kpis(self, results_path):
        valncreate_dir(results_path)
        data = pd.DataFrame(
            {
                "Epoch": np.arange(len(self._train_loss)),
                "Train Loss": self._train_loss,
                "Train LM Accuracy": self._train_lm_acc,
                "Val LM Accuracy": self._lm_acc,
                "Val MTB Loss": self._mtb_bce,
            }
        )
        data.to_csv(
            os.path.join(
                results_path, "kpis_{0}.csv".format(self.transformer)
            ),
            index=False,
        )
        return data

    def _train_epoch(
        self, epoch, update_size, save_best_model_only: bool = False
    ):
        start_time = super()._prepare_epoch(epoch)

        _train_acc, train_loss, train_mtb_bce = MTBModel._reset_train_metrics()
        train_lm_acc, train_mtb_bce_batch = [], []

        for i, data in enumerate(tqdm(self.data_loader.train_generator)):
            sequence, masked_label, e1_e2_start, blank_labels = data
            res = self._train_on_batch(
                sequence, masked_label, e1_e2_start, blank_labels
            )
            if res[0]:
                train_loss.append(res[0])
                train_lm_acc.append(res[1])
                train_mtb_bce.append(res[2])
            if (i % update_size) == (update_size - 1):
                logger.info(
                    f"{i+1}/{self.train_len} pools: - "
                    + f"Train loss: {np.mean(train_loss)}, "
                    + f"Train LM accuracy: {np.mean(train_lm_acc)}, "
                    + f"Train MTB Binary Cross Entropy {np.mean(train_mtb_bce)}"
                )

        self._train_loss.append(np.mean(train_loss))
        self._train_lm_acc.append(np.mean(train_lm_acc))

        self.on_epoch_end(
            epoch, self._mtb_bce, self._best_mtb_bce, save_best_model_only
        )

        logger.info("Train Loss: {0}".format(self._train_loss[-1]))
        logger.info("Train LM Accuracy: {0}".format(self._train_lm_acc[-1]))
        logger.info("Validation LM Accuracy: {0}".format(self._lm_acc[-1]))
        logger.info(
            "Validation MTB Binary Cross Entropy: {0}".format(
                self._mtb_bce[-1]
            )
        )

        logger.info(
            "Epoch finished, took {0} seconds.".format(
                time.time() - start_time
            )
        )

    def on_epoch_end(
        self, epoch, benchmark, baseline, save_best_model_only: bool = False
    ):
        """
        Function to run at the end of an epoch.

        Runs the evaluation method, increments the scheduler, sets a new baseline and appends the KPIS.ä

        Args:
            epoch: Current epoch
            benchmark: List of benchmark results
            baseline: Current baseline. Best model performance so far
        """
        eval_result = super().on_epoch_end(epoch, benchmark, baseline)
        self._best_mtb_bce = (
            eval_result[1]
            if eval_result[1] < self._best_mtb_bce
            else self._best_mtb_bce
        )
        self._mtb_bce.append(eval_result[1])
        self._lm_acc.append(eval_result[0])
        super().save_on_epoch_end(
            self._mtb_bce, self._best_mtb_bce, epoch, save_best_model_only
        )

    @classmethod
    def _reset_train_metrics(cls):
        train_loss, train_acc = super()._reset_train_metrics()
        train_mtb_bce = []
        return train_acc, train_loss, train_mtb_bce

    def _train_on_batch(
        self,
        sequence,
        mskd_label,
        e1_e2_start,
        blank_labels,
    ):
        mskd_label = mskd_label[(mskd_label != self.tokenizer.pad_token_id)]
        if mskd_label.shape[0] == 0:
            return None, None, None
        if self.train_on_gpu:
            mskd_label = mskd_label.cuda()
        blanks_logits, lm_logits = self._get_logits(e1_e2_start, sequence)
        loss = self.criterion(
            lm_logits,
            blanks_logits,
            mskd_label,
            blank_labels,
        )
        loss.backward()
        self._points_seen += len(sequence)
        clip_grad_norm_(self.model.parameters(), self.config.get("max_norm"))
        if self._points_seen > self.config.get("batch_size"):
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._points_seen = 0
        train_metrics = self.calculate_metrics(
            lm_logits,
            blanks_logits,
            mskd_label,
            blank_labels,
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
                    lm_logits,
                    blanks_logits,
                    masked_label,
                    blank_labels,
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
        self,
        lm_logits,
        blanks_logits,
        masked_for_pred,
        blank_labels,
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
