import logging
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from logger import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")


class MTBGenerator(Dataset):
    def __init__(self, data, tokenizer, dataset: str, max_size: int = None):
        """
        Data Generator for Matching the blanks models.

        Args:
            data: Dataset containing information about the relations and the
                position of the entities and the dataset.
            tokenizer: Huggingface transformers tokenizer to use
            dataset: Dataset type of the generator. May be train, validation or
                test,
            max_size: Maximum size of the batch.
        """

        self.entities_pools = [
            ep for ep in data["entities_pools"] if ep["set"] == dataset
        ]
        self.tokenized_relations = data["tokenized_relations"]
        self.n_relations = len(self.tokenized_relations)
        self.all_relation_ids = self.tokenized_relations[
            "relation_id"
        ].to_list()
        self.e1_pool = data["e1_pool"]
        self.e2_pool = data["e2_pool"]

        self.cls_idx = tokenizer.cls_token_id
        self.sep_idx = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_idx = tokenizer.mask_token_id

        self.blank_idx = tokenizer.convert_tokens_to_ids("[BLANK]")

        self.e1_idx = tokenizer.convert_tokens_to_ids("[E1]")
        self.e1e_idx = tokenizer.convert_tokens_to_ids("[/E1]")

        self.e2_idx = tokenizer.convert_tokens_to_ids("[E2]")
        self.e2e_idx = tokenizer.convert_tokens_to_ids("[/E2]")

        self.max_size = max_size

    def __iter__(self):
        """
        Create a generator that iterate over the Sequence.
        """
        idx = list(range(len(self)))
        random.shuffle(idx)
        yield from (item for item in (self[i] for i in idx))  # noqa: WPS335

    def __len__(self):
        return len(self.entities_pools) - 1

    def _put_blanks(self, data):
        alpha = 0.7
        r, e1, e2 = data
        blank_e1, blank_e2 = np.random.uniform(0, 1, 2)
        r0, r1, r2 = r
        r0 = np.array(r0)
        if blank_e1 < alpha:
            if r1[1] > r1[0]:
                r0 = np.append(
                    np.append(r0[: r1[0]], self.blank_idx), r0[r1[1] + 1 :]
                )
                diff = r1[1] - r1[0]
                r2 = (r2[0] - diff, r2[1] - diff)
                r1 = (r1[0], r1[0])
            else:
                r0[r1[0] : (r1[1] + 1)] = self.blank_idx
            e1 = "[BLANK]"

        if blank_e2 < alpha:
            if r2[1] > r2[0]:
                r0 = np.append(
                    np.append(r0[: r2[0]], self.blank_idx), r0[r2[1] + 1 :]
                )
                r2 = (r2[0], r2[0])
            else:
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
        pool = self.entities_pools[pool_id]
        positives = list(pool["relation_ids"])
        n_positives = (
            min(self.max_size, len(positives))
            if self.max_size
            else len(positives)
        )
        pos_idxs = random.sample(positives, n_positives)
        pos_df = self.tokenized_relations.iloc[pos_idxs]

        neg_idxs = self._sample_negative_indices(pool, pos_idxs)
        neg_df = self.tokenized_relations.loc[neg_idxs]

        batch = []
        batch = self._fill_batch_from_data(batch, pos_df, True)
        batch = self._fill_batch_from_data(batch, neg_df, False)

        return self._wrap_batch(batch)

    def _fill_batch_from_data(self, batch, data, positive: bool):
        for _idx, row in data.iterrows():
            e1_e2_start, masked_for_pred, x = self._preprocess(row)
            batch.append(
                (
                    x,
                    masked_for_pred,
                    e1_e2_start,
                    torch.LongTensor([int(positive)]),
                )
            )
        return batch

    def _sample_negative_indices(self, pool, pos_idxs):
        e1 = pool["e1"]
        e2 = pool["e2"]
        e1_represent = set(self.e1_pool[e1])
        e2_represent = set(self.e2_pool[e2])
        e1_negatives = e1_represent.difference(e2_represent)
        e2_negatives = e2_represent.difference(e1_represent)
        neg_idxs = None
        if np.random.uniform() > 0.5:
            if np.random.uniform() > 0.5:
                negatives = e1_negatives
            else:
                negatives = e2_negatives
            n_negatives = (
                min(self.max_size, len(negatives))
                if self.max_size
                else len(negatives)
            )
            neg_idxs = random.sample(negatives, n_negatives)
        if not neg_idxs:
            n_negatives = min(self.max_size, self.n_relations)
            neg_idx = [
                int(self.n_relations * random.random())
                for _ in range(n_negatives)
            ]
            neg_idxs = [self.all_relation_ids[p] for p in neg_idx]
            while any(n in pos_idxs for n in neg_idxs):
                neg_idx = [
                    int(self.n_relations * random.random())
                    for _ in range(n_negatives)
                ]
                neg_idxs = [self.all_relation_ids[p] for p in neg_idx]
        return neg_idxs

    def _wrap_batch(self, batch):
        sequences = [x[0] for x in batch]
        sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        mask_for_pred = list(map(lambda x: x[1], batch))
        mask_for_pred = pad_sequence(
            mask_for_pred,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        e1_e2_start = torch.stack(list(map(lambda x: x[2], batch)))
        labels = torch.stack(list(map(lambda x: x[3], batch)))
        return sequences, mask_for_pred, e1_e2_start, labels

    def _preprocess(self, row):
        r, e1, e2, relation_id = row
        r, e1, e2 = self._put_blanks((r, e1, e2))
        x, masked_for_pred, e1_e2_start = self._mask_sequence((r, e1, e2))
        x = torch.LongTensor(x)
        masked_for_pred = torch.LongTensor(masked_for_pred)
        e1_e2_start = torch.tensor(e1_e2_start)
        return e1_e2_start, masked_for_pred, x
