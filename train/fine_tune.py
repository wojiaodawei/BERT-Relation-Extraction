import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ml_utils.common import valncreate_dir
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from dataloaders.SemEvalDataloader import SemEvalDataloader
from model.albert.albert import AlbertModel
from model.bert.bert import BertModel

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")


class FineTuner:
    def __init__(self, config: dict):
        self.gradient_acc_steps = config.get("gradient_acc_steps")
        self.transformer = config.get("transformer")
        self.config = config
        self.data_loader = SemEvalDataloader(self.config)
        logger.info(
            "Loaded {0} fine-tuning samples.".format(
                self.data_loader.train_len
            )
        )

        self.tokenizer = self.data_loader.tokenizer
        e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        if e1_id == e2_id == 1:
            raise ValueError("e1_id == e2_id == 1")

        if "albert" in self.config.get("transformer"):
            self.model = AlbertModel.from_pretrained(
                force_download=False,
                model_size=self.config.get("transformer"),
                task="classification",
                n_classes=self.data_loader.n_classes,
                pretrained_model_name_or_path=self.config.get("transformer"),
            )

        else:
            self.model = BertModel.from_pretrained(
                model_size=self.config.get("transformer"),
                force_download=False,
                pretrained_model_name_or_path=self.config.get("transformer"),
                task="classification",
                n_classes=self.data_loader.n_classes,
            )
        logger.info("Freezing hidden layers")
        unfrozen_layers = [
            "classifier",
            "pooler",
            "encoder.layer.11",
            "classification_layer",
            "blanks_linear",
            "lm_linear",
            "cls",
        ]
        if "albert" in self.config.get("transformer"):
            unfrozen_layers = [
                "classifier",
                "pooler",
                "classification_layer",
                "blanks_linear",
                "lm_linear",
                "cls",
                "albert_layer_groups.0.albert_layers.0.ffn",
            ]
        for name, param in self.model.named_parameters():
            if not any([layer in name for layer in unfrozen_layers]):
                logger.info("FROZE {0}".format(name))
                param.requires_grad = False
            else:
                logger.info("FREE {0}".format(name))
                param.requires_grad = True
        self.model.resize_token_embeddings(len(self.tokenizer))
        pretrained_mtb_model = self.config.get("pretrained_mtb_model", None)
        if os.path.isfile(pretrained_mtb_model):
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
            del checkpoint, pretrained_dict, model_dict

        self.train_on_gpu = torch.cuda.is_available()
        if self.train_on_gpu:
            self.model.cuda()

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.optimizer = optim.Adam(
            [{"params": self.model.parameters(), "lr": self.config.get("lr")}]
        )

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30],
            gamma=0.8,
        )

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

    def train(self):
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
        update_size = len(self.data_loader.train_loader) // 10
        for epoch in range(self._start_epoch, self.config.get("epochs")):
            start_time = time.time()
            self.model.train()
            train_loss = 0.0
            train_loss_per_batch = []
            train_acc = 0.0
            train_acc_per_batch = []
            for i, data in enumerate(self.data_loader.train_loader, 0):
                x, e1_e2_start, labels, _, _, _ = data
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
                loss = loss / self.config.get("gradient_acc_steps")
                loss.backward()
                clip_grad_norm_(
                    self.model.parameters(), self.config.get("max_norm")
                )

                if (i % self.config.get("gradient_acc_steps")) == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss += loss.item()
                train_acc += FineTuner.evaluate_(
                    classification_logits, labels, ignore_idx=-1
                )[0]

                if (i % update_size) == (update_size - 1):
                    train_loss_per_batch.append(
                        self.config.get("gradient_acc_steps")
                        * train_loss
                        / update_size
                    )
                    train_acc_per_batch.append(train_acc / update_size)
                    print(
                        "[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f"
                        % (
                            epoch + 1,
                            (i + 1) * self.config.get("batch_size"),
                            self.data_loader.train_len,
                            train_loss_per_batch[-1],
                            train_acc_per_batch[-1],
                        )
                    )
                    train_loss = 0.0
                    train_acc = 0.0

            self.scheduler.step()
            self._train_loss.append(np.mean(train_loss_per_batch))
            self._train_acc.append(np.mean(train_acc_per_batch))

            results = self.evaluate_results()
            self._test_f1.append(results["f1"])
            self._test_acc.append(results["accuracy"])
            print(
                "Epoch finished, took %.2f seconds."
                % (time.time() - start_time)
            )
            print("Loss at Epoch %d: %.7f" % (epoch + 1, self._train_loss[-1]))
            print(
                "Train accuracy at Epoch %d: %.7f"
                % (epoch + 1, self._train_acc[-1])
            )
            print("Test f1 at Epoch %d: %.7f" % (epoch + 1, self._test_f1[-1]))
            print(
                "Test accuracy at Epoch %d: %.7f"
                % (epoch + 1, self._test_f1[-1])
            )

            if self._test_f1[-1] > self._best_test_f1:
                self._best_test_f1 = self._test_f1[-1]
                self._save_model(self.checkpoint_dir, epoch, best_model=True)

            self._save_model(
                self.checkpoint_dir, epoch,
            )

        logger.info("Finished Training.")

        results_path = os.path.join("results", "sem_eval")
        valncreate_dir(results_path)
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.scatter([e for e in range(len(self._train_loss))], self._train_loss)
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
        ax2.scatter([e for e in range(len(self._train_acc))], self._train_acc)
        ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax2.set_xlabel("Epoch", fontsize=22)
        ax2.set_ylabel("Training Accuracy", fontsize=22)
        ax2.set_title("Training Accuracy vs Epoch", fontsize=32)
        plt.savefig(
            os.path.join(
                results_path, "train_acc_{0}.png".format(self.transformer)
            )
        )

        fig3 = plt.figure(figsize=(20, 20))
        ax3 = fig3.add_subplot(111)
        ax3.scatter([e for e in range(len(self._test_f1))], self._test_f1)
        ax3.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax3.set_xlabel("Epoch", fontsize=22)
        ax3.set_ylabel("Test F1", fontsize=22)
        ax3.set_title("Test F1 vs Epoch", fontsize=32)
        plt.savefig(
            os.path.join(
                results_path, "test_f1_{0}.png".format(self.transformer)
            )
        )

        fig4 = plt.figure(figsize=(20, 20))
        ax4 = fig4.add_subplot(111)
        ax4.scatter([e for e in range(len(self._test_acc))], self._test_acc)
        ax4.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax4.set_xlabel("Epoch", fontsize=22)
        ax4.set_ylabel("Test Accuracy", fontsize=22)
        ax4.set_title("Test Accuracy vs Epoch", fontsize=32)
        plt.savefig(
            os.path.join(
                results_path, "test_f1_{0}.png".format(self.transformer)
            )
        )

        return self.model

    def load_best_model(self, checkpoint_dir: str):
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth.tar")
        logger.info("Loading best model from {0}".format(best_model_path))
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return (
            checkpoint["epoch"],
            checkpoint["best_f1"],
            checkpoint["losses_per_epoch"],
            checkpoint["accuracy_per_epoch"],
            checkpoint["test_acc"],
            checkpoint["test_f1"],
        )

    @classmethod
    def evaluate_(cls, output, labels, ignore_idx):
        idxs = (labels != ignore_idx).squeeze()
        o_labels = torch.softmax(output, dim=1).max(1)[1]
        l = labels.squeeze()[idxs]
        o = o_labels[idxs]

        if len(idxs) > 1:
            acc = (l == o).sum().item() / len(idxs)
        else:
            acc = (l == o).sum().item()
        l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
        o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

        return acc, (o, l)

    def evaluate_results(self):
        logger.info("Evaluating test samples")
        acc = 0
        out_labels = []
        true_labels = []
        self.model.eval()
        with torch.no_grad():
            for i, data in tqdm(
                enumerate(self.data_loader.test_loader),
                total=len(self.data_loader.test_loader),
            ):
                x, e1_e2_start, labels, _, _, _ = data
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

                accuracy, (o, l) = FineTuner.evaluate_(
                    classification_logits, labels, ignore_idx=-1
                )
                out_labels.append([str(i) for i in o])
                true_labels.append([str(i) for i in l])
                acc += accuracy

        accuracy = acc / (i + 1)
        results = {
            "accuracy": accuracy,
            "precision": precision_score(true_labels, out_labels),
            "recall": recall_score(true_labels, out_labels),
            "f1": f1_score(true_labels, out_labels),
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
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_f1": self._best_test_f1,
                "losses_per_epoch": self._train_loss,
                "accuracy_per_epoch": self._train_acc,
                "test_acc": self._test_acc,
                "test_f1": self._test_f1,
            },
            model_path,
        )
