import logging
import os
import re

import joblib
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AlbertTokenizer, BertTokenizer

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")


class SemEvalDataloader:
    def __init__(self, config: dict):
        """
        DataLoader for SemEval 2010 Task 8 data.

        Args:
            config: configuration parameters
        """
        self.config = config
        transformer = self.config.get("transformer")

        self.tokenizer = SemEvalDataloader.load_tokenizer(transformer)
        self.e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        self.n_classes = 0
        self.relations_mapper_path = "data/sem_eval/relations.pkl"
        self.trainset_path = "data/sem_eval/train.pkl"
        self.testset_path = "data/sem_eval/test.pkl"
        self.train_loader, self.test_loader = self.load_dataset()
        self.train_len = len(self.train_loader) * self.config.get("batch_size")
        self.test_len = len(self.test_loader) * self.config.get("batch_size")

    @classmethod
    def load_tokenizer(cls, transformer: str):
        """
        Loads the tokenizer based on th given transformer name.

        Args:
            transformer: Name of huggingface transformer
        """
        tokenizer_path = "data/{0}_tokenizer.pkl".format(transformer)
        if os.path.isfile(tokenizer_path):
            logger.info("Loading tokenizer from saved path.")
            with open(tokenizer_path, "rb") as pkl_file:
                return joblib.load(pkl_file)
        elif "albert" in transformer:
            tokenizer = AlbertTokenizer.from_pretrained(
                transformer, do_lower_case=False
            )
        else:
            tokenizer = BertTokenizer.from_pretrained(
                transformer, do_lower_case=False, add_special_tokens=True
            )
        tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]", "[BLANK]"])
        with open(tokenizer_path, "wb") as output:
            joblib.dump(tokenizer, output)

        logger.info(
            "Saved {0} tokenizer at {1}".format(transformer, tokenizer_path)
        )
        e1_id = tokenizer.convert_tokens_to_ids("[E1]")
        e2_id = tokenizer.convert_tokens_to_ids("[E2]")
        if e1_id == e2_id:
            raise ValueError("E1 token equals E2 token")
        return tokenizer

    def preprocess(self):
        """
        Data preprocessing for SemEval2010 task 8 dataset.
        """
        train_data_path = self.config.get("train_file")
        logger.info("Reading training file from {0}".format(train_data_path))
        with open(train_data_path, "r", encoding="utf8") as train_file:
            text = train_file.readlines()

        sents, relations, comments, blanks = self._preprocess_string(
            text, "train"
        )
        df_train = pd.DataFrame(data={"sents": sents, "relations": relations})

        test_data_path = self.config.get("test_file")
        logger.info("Reading test file from {0}".format(test_data_path))
        with open(test_data_path, "r", encoding="utf8") as test_file:
            text = test_file.readlines()

        (
            sents,
            relations,
            comments,
            blanks,
        ) = SemEvalDataloader._preprocess_string(text, "test")
        df_test = pd.DataFrame(data={"sents": sents, "relations": relations})

        rm = Relations_Mapper(df_train["relations"])
        with open(self.relations_mapper_path, "wb") as rm_output:
            joblib.dump(rm, rm_output)
        logger.info(
            "Saved relations mapper at {0}".format(self.relations_mapper_path)
        )

        df_train["relations_id"] = df_train.progress_apply(
            lambda x: rm.rel2idx[x["relations"]], axis=1
        )
        with open(self.trainset_path, "wb") as train_output:
            joblib.dump(df_train, train_output)
        logger.info("Saved trainset at {0}".format(self.trainset_path))

        df_test["relations_id"] = df_test.progress_apply(
            lambda x: rm.rel2idx[x["relations"]], axis=1
        )
        with open(self.testset_path, "wb") as test_output:
            joblib.dump(df_test, test_output)
        logger.info("Saved testset at {0}".format(self.testset_path))

        return df_train, df_test, rm

    @classmethod
    def _preprocess_string(cls, text, mode="train"):
        sents, relations, comments, blanks = [], [], [], []
        for i in range(int(len(text) / 4)):
            sent = text[4 * i]
            relation = text[4 * i + 1]
            comment = text[4 * i + 2]
            blank = text[4 * i + 3]

            # check entries
            if mode == "train" and not int(re.match("^\d+", sent)[0]) == (
                i + 1
            ):
                raise ValueError("No digit found")
            elif not (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1):
                raise ValueError("No digit found")
            if not re.match("^Comment", comment):
                raise ValueError("No comment found")
            if len(blank) != 1:
                raise ValueError("Too much blanks")

            sent = re.findall('"(.+)"', sent)[0]
            sent = re.sub("<e1>", "[E1]", sent)
            sent = re.sub("</e1>", "[/E1]", sent)
            sent = re.sub("<e2>", "[E2]", sent)
            sent = re.sub("</e2>", "[/E2]", sent)
            sents.append(sent)
            relations.append(relation), comments.append(comment)
            blanks.append(blank)
        return sents, relations, comments, blanks

    def load_dataset(self):
        if os.path.isfile(self.trainset_path) and os.path.isfile(
            self.testset_path
        ):
            df_train = joblib.load(self.trainset_path)
            df_test = joblib.load(self.testset_path)
            logger.info("Loaded preproccessed data.")
        else:
            df_train, df_test, rm = self.preprocess()

        self.n_classes = len(df_train["relations"].unique())
        train_set = semeval_dataset(
            df_train,
            tokenizer=self.tokenizer,
            e1_id=self.e1_id,
            e2_id=self.e2_id,
        )
        test_set = semeval_dataset(
            df_test,
            tokenizer=self.tokenizer,
            e1_id=self.e1_id,
            e2_id=self.e2_id,
        )
        PS = Pad_Sequence(
            seq_pad_value=self.tokenizer.pad_token_id,
            label_pad_value=self.tokenizer.pad_token_id,
            label2_pad_value=-1,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.get("batch_size"),
            shuffle=True,
            num_workers=0,
            collate_fn=PS,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.config.get("batch_size"),
            shuffle=True,
            num_workers=0,
            collate_fn=PS,
            pin_memory=False,
        )

        return train_loader, test_loader


class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}

        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1

        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key


class semeval_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Tokenizing data...")
        self.df["input"] = self.df.progress_apply(
            lambda x: tokenizer.encode(x["sents"]), axis=1
        )

        self.df["e1_e2_start"] = self.df.progress_apply(
            lambda x: get_e1e2_start(
                x["input"], e1_id=self.e1_id, e2_id=self.e2_id
            ),
            axis=1,
        )
        print(
            "\nInvalid rows/total: %d/%d"
            % (df["e1_e2_start"].isnull().sum(), len(df))
        )
        self.df.dropna(axis=0, inplace=True)

    def __len__(
        self,
    ):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.df.iloc[idx]["input"]),
            torch.LongTensor(self.df.iloc[idx]["e1_e2_start"]),
            torch.LongTensor([self.df.iloc[idx]["relations_id"]]),
        )


def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = (
            [i for i, e in enumerate(x) if e == e1_id][0],
            [i for i, e in enumerate(x) if e == e2_id][0],
        )
    except Exception as e:
        e1_e2_start = None
        print(e)
    return e1_e2_start


class Pad_Sequence:
    """
    collate_fn for dataloader to collate sequences of different lengths into a
    fixed length batch Returns padded x sequence, y sequence, x lengths and y
    lengths of batch.
    """

    def __init__(
        self,
        seq_pad_value,
        label_pad_value=-1,
        label2_pad_value=-1,
    ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value

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

        return (
            seqs_padded,
            labels_padded,
            labels2_padded,
            x_lengths,
            y_lengths,
            y2_lengths,
        )
