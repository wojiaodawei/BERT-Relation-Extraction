import logging
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from logger import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")


class SemEvalGenerator(Dataset):
    def __init__(self, data, tokenizer, batch_size):
        """
        Data Generator for SemEval 2010 Task 8.

        Args:
            data: Dataset containing information about the relations and the position of the entities
            tokenizer: Huggingface transformers tokenizer to use
            batch_size: Size of the final batches
        """
        self.tokenizer = tokenizer
        self.data = data.to_dict()
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data["sequence"]) // self.batch_size

    def __iter__(self):
        """
        Create a generator that iterate over the Sequence.
        """
        idx = list(range(len(self)))
        random.shuffle(idx)
        yield from (item for item in (self[i] for i in idx))  # noqa: WPS335

    def __getitem__(self, idx):
        batch = []

        for j in range(self.batch_size):
            batch.append(
                (
                    torch.LongTensor(self.data.get("sequence")[idx + j]),
                    torch.LongTensor(self.data.get("e1_e2_start")[idx + j]),
                    torch.LongTensor([self.data.get("relation_id")[idx + j]]),
                )
            )
        return self._wrap_batch(batch)

    def _wrap_batch(self, batch):
        sequences = [x[0] for x in batch]
        sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        e1_e2_start = list(map(lambda x: x[1], batch))
        e1_e2_start = pad_sequence(
            e1_e2_start, batch_first=True, padding_value=-1
        )
        labels = list(map(lambda x: x[2], batch))
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        return (
            sequences,
            e1_e2_start,
            labels,
        )
