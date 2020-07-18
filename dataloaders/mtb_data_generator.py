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
            ep for ep in self.data["entities_pools"] if ep[3] == dataset
        ]
        self.tokenizer = tokenizer

        self.r_entities_map = {}
        for idx, class_pool in enumerate(self.data["entities_pools"]):
            for q in class_pool[0]:
                self.r_entities_map[q] = idx

        self.r_indices = list(self.r_entities_map.keys())
        self.total_data_size = len(self.r_indices)

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
        yield from (item for item in [self[i] for i in range(len(self))])

    def __len__(self):
        return len(self.r_entities_map) - 1

    def _put_blanks(self, data):
        alpha = 0.7
        r, e1, e2 = data
        blank_e1 = np.random.uniform()
        blank_e2 = np.random.uniform()
        r0, r1, r2 = r
        r0 = np.array(r0)
        if blank_e1 >= alpha:
            r0[np.array(r1)] = self.blank_idx
            e1 = "[BLANK]"

        if blank_e2 >= alpha ** 2:
            r0[np.array(r2)] = self.blank_idx
            e2 = "[BLANK]"
        r = (r0, r1, r2)
        return (r, e1, e2)

    def _mask_sequence(self, data):
        mask_probability = 0.15
        (x, s1, s2), e1, e2 = data

        forbidden_idxs = [min(s1) - 1] + np.unique(s1).tolist() + [max(s1) + 1]
        forbidden_idxs += (
            [min(s2) - 1] + np.unique(s2).tolist() + [max(s2) + 1]
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

    def __getitem__(self, idx):
        tokenized_relations = self.data["tokenized_relations"]
        idx = list(self.r_entities_map.keys())[idx]
        pool_id = self.r_entities_map[idx]
        this_entity_pool = self.data["entities_pools"][pool_id]
        pos_idxs = np.random.choice(
            this_entity_pool[0],
            size=min(int(self.batch_size // 2), len(this_entity_pool[0])),
            replace=False,
        )
        non_easy_negatives = set(
            this_entity_pool[0] + this_entity_pool[1] + this_entity_pool[2]
        )
        negatives = set(tokenized_relations["relation_id"]).difference(
            non_easy_negatives
        )
        if np.random.uniform() > 0.5:  # Sample hard negatives
            if np.random.uniform() > 0.5:  # e2 negatives
                negatives = this_entity_pool[1]

            else:  # e1 negatives
                negatives = this_entity_pool[2]
        if not negatives:
            negatives = set(tokenized_relations["relation_id"]).difference(
                non_easy_negatives
            )
        neg_idxs = np.random.choice(
            list(negatives),
            size=min(int(self.batch_size // 2), len(negatives)),
            replace=False,
        )
        q = 1 / len(negatives)

        batch = []

        # process positive sample
        pos_df = tokenized_relations.iloc[pos_idxs]
        for _pos_idx, pos_row in pos_df.iterrows():
            e1_e2_start, masked_for_pred, x = self._preprocess(pos_row)
            batch.append(
                (
                    x,
                    masked_for_pred,
                    e1_e2_start,
                    torch.FloatTensor([1.0]),
                    torch.LongTensor([1]),
                )
            )

        # process negative samples
        negs_df = tokenized_relations.loc[neg_idxs]
        for _neg_idx, neg_row in negs_df.iterrows():
            e1_e2_start, masked_for_pred, x = self._preprocess(neg_row)
            batch.append(
                (
                    x,
                    masked_for_pred,
                    e1_e2_start,
                    torch.FloatTensor([q]),
                    torch.LongTensor([0]),
                )
            )

        return self._wrap_batch(batch)

    def _wrap_batch(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(
            seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(
            labels2, batch_first=True, padding_value=-1
        )
        labels4 = list(map(lambda x: x[4], sorted_batch))
        labels4_padded = pad_sequence(
            labels4, batch_first=True, padding_value=-1
        )
        return (
            seqs_padded,
            labels_padded,
            labels2_padded,
            labels4_padded,
        )

    def _preprocess(self, row):
        r, e1, e2, relation_id = row
        r, e1, e2 = self._put_blanks((r, e1, e2))
        x, masked_for_pred, e1_e2_start = self._mask_sequence((r, e1, e2))
        x = torch.LongTensor(x)
        masked_for_pred = torch.LongTensor(masked_for_pred)
        e1_e2_start = torch.tensor(e1_e2_start)
        return e1_e2_start, masked_for_pred, x
