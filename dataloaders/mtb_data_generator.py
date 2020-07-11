import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AlbertTokenizer

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")


class MTBTrainGenerator(Dataset):
    def __init__(self, dataset, batch_size=None):
        self.batch_size = batch_size
        self.alpha = 0.7
        self.mask_probability = 0.15

        self.df = pd.DataFrame(dataset, columns=["r", "e1", "e2"])

        tokenizer_path = "data/ALBERT_tokenizer.pkl"
        if os.path.isfile(tokenizer_path):
            with open(tokenizer_path, "rb") as pkl_file:
                self.tokenizer = joblib.load(pkl_file)
            logger.info("Loaded tokenizer from saved path.")
        else:
            self.tokenizer = AlbertTokenizer.from_pretrained(
                "albert-large-v2", do_lower_case=False
            )
            self.tokenizer.add_tokens(
                ["[E1]", "[/E1]", "[E2]", "[/E2]", "[BLANK]"]
            )
            with open(tokenizer_path, "wb") as output:
                joblib.dump(self.tokenizer, output)

            logger.info("Saved ALBERT tokenizer at {0}".format(tokenizer_path))
        e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        if not e1_id != e2_id != 1:
            raise ValueError("E1 token == E2 token == 1")

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.E1_token_id = self.tokenizer.encode("[E1]")[1:-1][0]
        self.E1s_token_id = self.tokenizer.encode("[/E1]")[1:-1][0]
        self.E2_token_id = self.tokenizer.encode("[E2]")[1:-1][0]
        self.E2s_token_id = self.tokenizer.encode("[/E2]")[1:-1][0]
        self.PS = Pad_Sequence(
            seq_pad_value=self.tokenizer.pad_token_id,
            label_pad_value=self.tokenizer.pad_token_id,
            label2_pad_value=-1,
            label3_pad_value=-1,
            label4_pad_value=-1,
        )

    def put_blanks(self, d):
        blank_e1 = np.random.uniform()
        blank_e2 = np.random.uniform()
        if blank_e1 >= self.alpha:
            r, e1, e2 = d
            d = (r, "[BLANK]", e2)

        if blank_e2 >= self.alpha:
            r, e1, e2 = d
            d = (r, e1, "[BLANK]")
        return d

    def tokenize(self, D):
        (x, s1, s2), e1, e2 = D
        x = [
            w.lower() for w in x if x != "[BLANK]"
        ]  # we are using uncased model

        ### Include random masks for MLM training
        forbidden_idxs = [i for i in range(s1[0], s1[1])] + [
            i for i in range(s2[0], s2[1])
        ]
        pool_idxs = [i for i in range(len(x)) if i not in forbidden_idxs]
        masked_idxs = np.random.choice(
            pool_idxs,
            size=round(self.mask_probability * len(pool_idxs)),
            replace=False,
        )
        masked_for_pred = [
            token.lower()
            for idx, token in enumerate(x)
            if (idx in masked_idxs)
        ]
        # masked_for_pred = [w.lower() for w in masked_for_pred] # we are using uncased model
        x = [
            token if (idx not in masked_idxs) else self.tokenizer.mask_token
            for idx, token in enumerate(x)
        ]

        ### replace x spans with '[BLANK]' if e is '[BLANK]'
        if (e1 == "[BLANK]") and (e2 != "[BLANK]"):
            x = (
                [self.cls_token]
                + x[: s1[0]]
                + ["[E1]", "[BLANK]", "[/E1]"]
                + x[s1[1] : s2[0]]
                + ["[E2]"]
                + x[s2[0] : s2[1]]
                + ["[/E2]"]
                + x[s2[1] :]
                + [self.sep_token]
            )

        elif (e1 == "[BLANK]") and (e2 == "[BLANK]"):
            x = (
                [self.cls_token]
                + x[: s1[0]]
                + ["[E1]", "[BLANK]", "[/E1]"]
                + x[s1[1] : s2[0]]
                + ["[E2]", "[BLANK]", "[/E2]"]
                + x[s2[1] :]
                + [self.sep_token]
            )

        elif (e1 != "[BLANK]") and (e2 == "[BLANK]"):
            x = (
                [self.cls_token]
                + x[: s1[0]]
                + ["[E1]"]
                + x[s1[0] : s1[1]]
                + ["[/E1]"]
                + x[s1[1] : s2[0]]
                + ["[E2]", "[BLANK]", "[/E2]"]
                + x[s2[1] :]
                + [self.sep_token]
            )

        elif (e1 != "[BLANK]") and (e2 != "[BLANK]"):
            x = (
                [self.cls_token]
                + x[: s1[0]]
                + ["[E1]"]
                + x[s1[0] : s1[1]]
                + ["[/E1]"]
                + x[s1[1] : s2[0]]
                + ["[E2]"]
                + x[s2[0] : s2[1]]
                + ["[/E2]"]
                + x[s2[1] :]
                + [self.sep_token]
            )

        e1_e2_start = (
            [i for i, e in enumerate(x) if e == "[E1]"][0],
            [i for i, e in enumerate(x) if e == "[E2]"][0],
        )

        x = self.tokenizer.convert_tokens_to_ids(x)
        masked_for_pred = self.tokenizer.convert_tokens_to_ids(masked_for_pred)
        return x, masked_for_pred, e1_e2_start

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r, e1, e2 = self.df.iloc[idx]  # positive sample
        pool = self.df[((self.df["e1"] == e1) & (self.df["e2"] == e2))].index
        pool = pool.append(
            self.df[((self.df["e1"] == e2) & (self.df["e2"] == e1))].index
        )
        pos_idxs = np.random.choice(
            pool,
            size=min(int(self.batch_size // 2), len(pool)),
            replace=False,
        )
        ### get negative samples
        """
        choose from option: 
        1) sampling uniformly from all negatives
        2) sampling uniformly from negatives that share e1 or e2
        """
        if np.random.uniform() > 0.5:
            pool = self.df[
                ((self.df["e1"] != e1) | (self.df["e2"] != e2))
            ].index
            neg_idxs = np.random.choice(
                pool,
                size=min(int(self.batch_size // 2), len(pool)),
                replace=False,
            )
            Q = 1 / len(pool)

        else:
            if np.random.uniform() > 0.5:  # share e1 but not e2
                pool = self.df[
                    ((self.df["e1"] == e1) & (self.df["e2"] != e2))
                ].index
                if len(pool) > 0:
                    neg_idxs = np.random.choice(
                        pool,
                        size=min(int(self.batch_size // 2), len(pool)),
                        replace=False,
                    )
                else:
                    neg_idxs = []

            else:  # share e2 but not e1
                pool = self.df[
                    ((self.df["e1"] != e1) & (self.df["e2"] == e2))
                ].index
                if len(pool) > 0:
                    neg_idxs = np.random.choice(
                        pool,
                        size=min(int(self.batch_size // 2), len(pool)),
                        replace=False,
                    )
                else:
                    neg_idxs = []

            if len(neg_idxs) == 0:  # if empty, sample from all negatives
                pool = self.df[
                    ((self.df["e1"] != e1) | (self.df["e2"] != e2))
                ].index
                neg_idxs = np.random.choice(
                    pool,
                    size=min(int(self.batch_size // 2), len(pool)),
                    replace=False,
                )
            Q = 1 / len(pool)

        batch = []
        ## process positive sample
        pos_df = self.df.loc[pos_idxs]
        for idx, row in pos_df.iterrows():
            r, e1, e2 = row[0], row[1], row[2]
            x, masked_for_pred, e1_e2_start = self.tokenize(
                self.put_blanks((r, e1, e2))
            )
            x = torch.LongTensor(x)
            masked_for_pred = torch.LongTensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            # e1, e2 = torch.tensor(e1), torch.tensor(e2)
            batch.append(
                (
                    x,
                    masked_for_pred,
                    e1_e2_start,
                    torch.FloatTensor([1.0]),
                    torch.LongTensor([1]),
                )
            )

        ## process negative samples
        negs_df = self.df.loc[neg_idxs]
        for idx, row in negs_df.iterrows():
            r, e1, e2 = row[0], row[1], row[2]
            x, masked_for_pred, e1_e2_start = self.tokenize(
                self.put_blanks((r, e1, e2))
            )
            x = torch.LongTensor(x)
            masked_for_pred = torch.LongTensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            # e1, e2 = torch.tensor(e1), torch.tensor(e2)
            batch.append(
                (
                    x,
                    masked_for_pred,
                    e1_e2_start,
                    torch.FloatTensor([Q]),
                    torch.LongTensor([0]),
                )
            )
        batch = self.PS(batch)
        return batch


class Pad_Sequence:
    """
    collate_fn for dataloader to collate sequences of different lengths into a
    fixed length batch Returns padded x sequence, y sequence, x lengths and y
    lengths of batch.
    """

    def __init__(
        self,
        seq_pad_value,
        label_pad_value=1,
        label2_pad_value=-1,
        label3_pad_value=-1,
        label4_pad_value=-1,
    ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        self.label3_pad_value = label3_pad_value
        self.label4_pad_value = label4_pad_value

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(
            seqs, batch_first=True, padding_value=self.seq_pad_value
        )
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_value
        )
        y_lengths = torch.LongTensor([len(x) for x in labels])

        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(
            labels2, batch_first=True, padding_value=self.label2_pad_value
        )
        y2_lengths = torch.LongTensor([len(x) for x in labels2])

        labels3 = list(map(lambda x: x[3], sorted_batch))
        labels3_padded = pad_sequence(
            labels3, batch_first=True, padding_value=self.label3_pad_value
        )
        y3_lengths = torch.LongTensor([len(x) for x in labels3])

        labels4 = list(map(lambda x: x[4], sorted_batch))
        labels4_padded = pad_sequence(
            labels4, batch_first=True, padding_value=self.label4_pad_value
        )
        y4_lengths = torch.LongTensor([len(x) for x in labels4])
        return (
            seqs_padded,
            labels_padded,
            labels2_padded,
            labels3_padded,
            labels4_padded,
            x_lengths,
            y_lengths,
            y2_lengths,
            y3_lengths,
            y4_lengths,
        )
