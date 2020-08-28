import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")


class MTBGenerator(Dataset):
    def __init__(self, data, tokenizer, dataset: str, batch_size=None):
        """
        Data Generator for Matching the blanks models.

        Args:
            data: Dataset containing information about the relations and the position of the entities and the dataset
            tokenizer: Huggingface transformers tokenizer to use
            dataset: Dataset type of the generator. May be train, validation or test
            batch_size: Batch size
        """
        self.batch_size = batch_size

        self.data = data
        self.data["entities_pools"] = [
            ep[:-1] for ep in self.data["entities_pools"] if ep[-1] == dataset
        ]
        self.tokenizer = tokenizer

        self.blank_idx = self.tokenizer.convert_tokens_to_ids("[BLANK]")
        self.mask_idx = self.tokenizer.mask_token_id

        self.e1_idx = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e1e_idx = self.tokenizer.convert_tokens_to_ids("[/E1]")

        self.e2_idx = self.tokenizer.convert_tokens_to_ids("[E2]")
        self.e2e_idx = self.tokenizer.convert_tokens_to_ids("[/E2]")

        self.cls_idx = self.tokenizer.cls_token_id
        self.sep_idx = self.tokenizer.sep_token_id

    def __iter__(self):
        """
        Create a generator that iterate over the Sequence.
        """
        yield from (
            item for item in (self[i] for i in range(len(self)))
        )  # noqa: WPS335

    def __len__(self):
        return len(self.data["entities_pools"]) - 1

    def _put_blanks(self, data):
        alpha = 0.7
        r, e1, e2 = data
        blank_e1 = np.random.uniform()
        blank_e2 = np.random.uniform()
        r0, r1, r2 = r
        r0 = np.array(r0)
        if blank_e1 < alpha:
            r0[r1[0] : (r1[1] + 1)] = self.blank_idx
            e1 = "[BLANK]"

        if blank_e2 < alpha:
            r0[r2[0] : (r2[1] + 1)] = self.blank_idx
            e2 = "[BLANK]"
        r = (r0, r1, r2)
        return (r, e1, e2)

    def _mask_sequence(self, data):
        mask_probability = 0.15
        (x, s1, s2), e1, e2 = data
        forbidden_idxs = set(np.arange(max(s1[0] - 1, 0), s1[1] + 2))
        forbidden_idxs = forbidden_idxs.union(
            set(np.arange(max(s2[0] - 1, 0), s2[1] + 2))
        )

        pool_idxs = [i for i in range(len(x)) if i not in forbidden_idxs]
        masked_idxs = np.random.choice(
            pool_idxs,
            size=round(mask_probability * len(pool_idxs)),
            replace=False,
        )
        masked_for_pred = [
            xi for idx, xi in enumerate(x) if (idx in masked_idxs)
        ]
        mask_label = [
            xi if (idx in masked_idxs) else 0 for idx, xi in enumerate(x)
        ]
        sequence = [
            self.mask_idx if mask else token
            for token, mask in zip(x, mask_label)
        ]
        e1_start = s1[0] - 1
        e2_start = s2[0] - 1
        entities_start = np.array([e1_start, e2_start])

        return sequence, masked_for_pred, entities_start

    def __getitem__(self, pool_id):
        tokenized_relations = self.data["tokenized_relations"]
        positives = self.data["entities_pools"][pool_id]
        e1 = set(self.data["tokenized_relations"].iloc[positives]["e1"])
        e1 = list(e1)[0]
        e2 = set(self.data["tokenized_relations"].iloc[positives]["e2"])
        e2 = list(e2)[0]

        e1_represent = set(self.data["e1_pool"][e1])
        e2_represent = set(self.data["e2_pool"][e2])

        e1_negatives = e1_represent.difference(e2_represent)
        e2_negatives = e2_represent.difference(e1_represent)

        pos_idxs = np.random.choice(
            positives,
            size=min(int(self.batch_size // 2), len(positives)),
            replace=False,
        )
        non_easy_negatives = e1_negatives.union(e2_negatives)

        negatives = set(tokenized_relations["relation_id"]).difference(
            non_easy_negatives
        )
        if np.random.uniform() > 0.5:  # Sample hard negatives
            if np.random.uniform() > 0.5:  # e2 negatives
                negatives = e1_negatives

            else:  # e1 negatives
                negatives = e2_negatives
        if not negatives:
            negatives = set(tokenized_relations["relation_id"]).difference(
                non_easy_negatives
            )
        neg_idxs = np.random.choice(
            list(negatives),
            size=min(int(self.batch_size // 2), len(negatives)),
            replace=False,
        )

        batch = []

        # process positive sample
        pos_df = tokenized_relations.iloc[pos_idxs]
        for _pos_idx, pos_row in pos_df.iterrows():
            e1_e2_start, masked_for_pred, x = self._preprocess(pos_row)
            batch.append(
                (x, masked_for_pred, e1_e2_start, torch.LongTensor([1]))
            )

        # process negative samples
        negs_df = tokenized_relations.loc[neg_idxs]
        for _neg_idx, neg_row in negs_df.iterrows():
            e1_e2_start, masked_for_pred, x = self._preprocess(neg_row)
            batch.append(
                (x, masked_for_pred, e1_e2_start, torch.LongTensor([0]))
            )

        return self._wrap_batch(batch)

    def _wrap_batch(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        mask_for_pred = list(map(lambda x: x[1], sorted_batch))
        mask_for_pred = pad_sequence(
            mask_for_pred,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        e1_e2_start = list(map(lambda x: x[2], sorted_batch))
        e1_e2_start = pad_sequence(
            e1_e2_start, batch_first=True, padding_value=-1
        )
        labels = list(map(lambda x: x[3], sorted_batch))
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        return (
            sequences,
            mask_for_pred,
            e1_e2_start,
            labels,
        )

    def _preprocess(self, row):
        r, e1, e2, relation_id = row
        r, e1, e2 = self._put_blanks((r, e1, e2))
        x, masked_for_pred, e1_e2_start = self._mask_sequence((r, e1, e2))
        x = torch.LongTensor(x)
        masked_for_pred = torch.LongTensor(masked_for_pred)
        e1_e2_start = torch.tensor(e1_e2_start)
        return e1_e2_start, masked_for_pred, x
