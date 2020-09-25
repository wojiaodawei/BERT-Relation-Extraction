import logging
import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from ml_utils.path_operations import valncreate_dir
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataloaders.semeval_dataloader import SemEvalDataloader
from logger import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from model.bert import BertModel
from model.relation_extractor import RelationExtractor

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")

sns.set(font_scale=2.2)


class SemEvalModel(RelationExtractor):
    def __init__(self, config: dict):
        """
        SemEval Model using Transformers.

        Args:
            config: configuration parameters
        """
        super().__init__()
        self.gradient_acc_steps = config.get("gradient_acc_steps")
        self.transformer = config.get("transformer")
        self.config = config
        self.data_loader = SemEvalDataloader(self.config)
        logger.info(
            f"Loaded {len(self.data_loader.train_generator)} fine-tuning samples."
        )

        self.tokenizer = self.data_loader.tokenizer
        self.tokenizer.convert_tokens_to_ids("[E1]")
        self.tokenizer.convert_tokens_to_ids("[E2]")

        self.model = BertModel.from_pretrained(
            model_size=self.config.get("transformer"),
            force_download=False,
            pretrained_model_name_or_path=self.config.get("transformer"),
            task="classification",
            n_classes=self.data_loader.n_classes,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        pretrained_mtb_model = self.config.get("pretrained_mtb_model", None)
        if pretrained_mtb_model and os.path.isfile(pretrained_mtb_model):
            self._load_pretrained_model(pretrained_mtb_model)

        self.train_on_gpu = torch.cuda.is_available() and config.get(
            "use_gpu", True
        )
        if self.train_on_gpu:
            self.model.cuda()

        self.criterion = CrossEntropyLoss(reduction="sum")

        self._start_epoch = 0
        self._train_loss = []
        self._train_acc = []
        self._test_f1 = []
        self._test_acc = []
        self._best_test_f1 = 0
        self.checkpoint_dir = os.path.join(
            "models", "finetuning", "sem_eval", self.transformer
        )
        valncreate_dir(self.checkpoint_dir)

        self._points_seen = 0

    def _load_pretrained_model(self, pretrained_mtb_model):
        logger.info(
            "Loading pre-trained MTB model from {0}.".format(
                pretrained_mtb_model
            )
        )
        checkpoint = torch.load(pretrained_mtb_model)
        model_dict = self.model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k in model_dict.keys()
        }
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.tokenizer = checkpoint["tokenizer"]

    def train(self, epochs):
        """
        Runs the training.
        """
        pretrained_model = self.config.get("pretrained_mtb_model", None)
        pretrained_model = (
            "pretrained" if pretrained_model else "no_pretraining"
        )
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.config.get("lr")
        )
        ovr_steps = (
            epochs
            * len(self.data_loader.train_generator)
            * self.config.get("mini_batch_size")
            / self.config.get("batch_size")
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, ovr_steps // 10, ovr_steps
        )

        results_path = os.path.join(
            "results", "sem_eval", pretrained_model, str(epochs)
        )
        best_model_path = os.path.join(
            self.checkpoint_dir, "best_model.pth.tar"
        )
        resume = self.config.get("resume", False)
        if resume and os.path.exists(best_model_path):
            (
                self._start_epoch,
                self._best_test_f1,
                self._train_loss,
                self._train_acc,
                self._test_f1,
                self._train_acc,
            ) = self.load_best_model(self.checkpoint_dir)

        logger.info("Starting training process")
        pad_id = self.tokenizer.pad_token_id
        for epoch in range(self._start_epoch, epochs):
            self._train_epoch(epoch, pad_id, optimizer, scheduler)
            data = self._write_kpis(results_path)
            self._plot_results(data, results_path)

        logger.info("Finished Training.")
        return self.model

    def _train_epoch(self, epoch, pad_id, optimizer, scheduler):
        start_time = super()._train_epoch(epoch)

        train_loss, train_acc = [], []

        for i, data in enumerate(tqdm(self.data_loader.train_generator)):
            x, e1_e2_start, labels = data
            classification_logits, labels, loss = self._train_on_batch(
                e1_e2_start, labels, pad_id, x, optimizer, scheduler
            )

            train_loss.append(loss)
            train_acc.append(
                SemEvalModel.evaluate_inner(classification_logits, labels)[0]
            )

        self._train_loss.append(np.mean(train_loss))
        self._train_acc.append(np.mean(train_acc))

        self.on_epoch_end(epoch, self._test_f1, self._best_test_f1)

        logger.info(f"Train Loss: {self._train_loss[-1]}")
        logger.info(f"Train Accuracy: {self._train_acc[-1]}")
        logger.info(f"Test Accuracy: {self._test_acc[-1]}")
        logger.info(f"Test F1: {self._test_f1[-1]}")

        logger.info(
            "Epoch finished, took {0} seconds.".format(
                time.time() - start_time
            )
        )

    def on_epoch_end(self, epoch, benchmark, baseline):
        """
        Function to run at the end of an epoch.

        Runs the evaluation method, increments the scheduler, sets a new baseline and appends the KPIS.ä

        Args:
            epoch: Current epoch
            benchmark: List of benchmark results
            baseline: Current baseline. Best model performance so far
        """
        eval_result = self.evaluate()
        self._best_test_f1 = (
            eval_result.get("f1")
            if eval_result.get("f1") > self._best_test_f1
            else self._best_test_f1
        )
        self._test_f1.append(eval_result["f1"])
        self._test_acc.append(eval_result["accuracy"])

    def _train_on_batch(
        self, e1_e2_start, labels, pad_id, x, optimizer, scheduler
    ):
        attention_mask = (x != pad_id).float()
        token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
        if self.train_on_gpu:
            x = x.cuda()
            labels = labels.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
        classification_logits = self.model(
            x,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            e1_e2_start=e1_e2_start,
        )
        loss = self.criterion(classification_logits, labels.squeeze(1))
        loss_p = loss.item()
        loss = loss / self.config.get("batch_size")
        loss.backward()
        self._points_seen += len(x)
        if self._points_seen >= self.config.get("batch_size"):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self._points_seen = 0
        return classification_logits, labels, loss_p / len(x)

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
        sns.lineplot(x="Epoch", y="Test F1", ax=ax, data=data, linewidth=4)
        ax.set_title("Test F1 Score")
        plt.savefig(
            os.path.join(save_at, "test_f1_{0}.png".format(self.transformer))
        )
        plt.close(fig)

        tmp = data[["Epoch", "Train Accuracy", "Test Accuracy"]].melt(
            id_vars="Epoch", var_name="Set", value_name="Accuracy"
        )
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.lineplot(
            x="Epoch",
            y="Accuracy",
            hue="Set",
            ax=ax,
            data=tmp,
            linewidth=4,
        )
        ax.set_title("Accuracy")
        plt.savefig(
            os.path.join(save_at, "acc_{0}.png".format(self.transformer))
        )
        plt.close(fig)

    def _write_kpis(self, results_path):
        valncreate_dir(results_path)
        data = pd.DataFrame(
            {
                "Epoch": np.arange(len(self._train_loss)),
                "Train Loss": self._train_loss,
                "Train Accuracy": self._train_acc,
                "Test Accuracy": self._test_acc,
                "Test F1": self._test_f1,
            }
        )
        data.to_csv(
            os.path.join(
                results_path, "kpis_{0}.csv".format(self.transformer)
            ),
            index=False,
        )
        return data

    def load_best_model(self, checkpoint_dir: str):
        """
        Loads the current best model in the checkpoint directory.

        Args:
            checkpoint_dir: Checkpoint directory path
        """
        checkpoint = super().load_best_model(checkpoint_dir)
        return (
            checkpoint["epoch"],
            checkpoint["best_f1"],
            checkpoint["losses_per_epoch"],
            checkpoint["accuracy_per_epoch"],
            checkpoint["test_acc"],
            checkpoint["test_f1"],
        )

    @classmethod
    def evaluate_inner(cls, output, labels):
        l = labels.squeeze()
        o = output.max(1)[1]
        return (l == o).float().mean().item(), (o, l)

    def evaluate(self):
        logger.info("Evaluating test samples")
        out_labels = []
        true_labels = []
        self.model.eval()
        with torch.no_grad():
            for i, data in tqdm(
                enumerate(self.data_loader.test_generator),
                total=len(self.data_loader.test_generator),
            ):
                x, e1_e2_start, labels = data
                attention_mask = (x != self.tokenizer.pad_token_id).float()
                token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

                if self.train_on_gpu:
                    x = x.cuda()
                    labels = labels.cuda()
                    attention_mask = attention_mask.cuda()
                    token_type_ids = token_type_ids.cuda()

                classification_logits = self.model(
                    x,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    e1_e2_start=e1_e2_start,
                )

                _acc, (o, l) = SemEvalModel.evaluate_inner(
                    classification_logits, labels
                )
                out_labels.extend([int(i) for i in o])
                true_labels.extend([int(i) for i in l])

        out_labels = np.array(out_labels)
        true_labels = np.array(true_labels)
        accuracy = np.mean([o == t for t, o in zip(out_labels, true_labels)])
        results = {
            "accuracy": accuracy,
            "precision": precision_score(
                true_labels, out_labels, average="micro"
            ),
            "recall": recall_score(true_labels, out_labels, average="micro"),
            "f1": f1_score(true_labels, out_labels, average="micro"),
        }
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

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
                "best_f1": self._best_test_f1,
                "losses_per_epoch": self._train_loss,
                "accuracy_per_epoch": self._train_acc,
                "test_acc": self._test_acc,
                "test_f1": self._test_f1,
            },
            model_path,
        )
