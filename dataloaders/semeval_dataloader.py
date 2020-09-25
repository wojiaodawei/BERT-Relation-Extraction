import logging
import os
import re

import joblib
import pandas as pd
from ml_utils.normalizer import Normalizer
from tqdm import tqdm
from transformers import AlbertTokenizer, BertTokenizer

from dataloaders.semeval_data_generator import SemEvalGenerator
from logger import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

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
        self.normalizer = Normalizer("en", config.get("normalization", []))

        self.n_classes = 0
        self.relations_mapper_path = "data/sem_eval/relations.pkl"
        self.trainset_path = "data/sem_eval/train.pkl"
        self.testset_path = "data/sem_eval/test.pkl"
        df_train, df_test = self.load_dataset()
        self.train_generator = SemEvalGenerator(
            data=df_train,
            tokenizer=self.tokenizer,
            batch_size=self.config.get("mini_batch_size"),
        )
        self.test_generator = SemEvalGenerator(
            data=df_test,
            tokenizer=self.tokenizer,
            batch_size=self.config.get("batch_size"),
        )

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
        return tokenizer

    def preprocess(self):
        """
        Data preprocessing for SemEval2010 task 8 dataset.
        """
        train_data_path = self.config.get("train_file")
        logger.info("Reading training file from {0}".format(train_data_path))
        with open(train_data_path, "r", encoding="utf8") as train_file:
            text = train_file.readlines()
        sents, relations = self._preprocess_string(text)
        e1_e2_start = []
        logger.info("Tokenizing train data")
        for trainset_idx, trainset_s in enumerate(tqdm(sents)):
            trainset_s = self.tokenize_dataset(e1_e2_start, trainset_s)
            sents[trainset_idx] = trainset_s
        df_train = pd.DataFrame(
            data={
                "sequence": sents,
                "relation": relations,
                "e1_e2_start": e1_e2_start,
            }
        )

        test_data_path = self.config.get("test_file")
        logger.info("Reading test file from {0}".format(test_data_path))
        with open(test_data_path, "r", encoding="utf8") as test_file:
            text = test_file.readlines()
        sents, relations = self._preprocess_string(text)
        e1_e2_start = []
        logger.info("Tokenizing test data")
        for testset_idx, testset_s in enumerate(tqdm(sents)):
            testset_s = self.tokenize_dataset(e1_e2_start, testset_s)
            sents[testset_idx] = testset_s
        df_test = pd.DataFrame(
            data={
                "sequence": sents,
                "relation": relations,
                "e1_e2_start": e1_e2_start,
            }
        )

        rm = RelationsMap(df_train["relation"])
        with open(self.relations_mapper_path, "wb") as rm_output:
            joblib.dump(rm, rm_output)
        logger.info(f"Saved relations map at {self.relations_mapper_path}")

        df_train["relation_id"] = df_train.progress_apply(
            lambda x: rm.rel2idx[x["relation"]], axis=1
        )
        df_train.drop(columns=["relation"], inplace=True)
        with open(self.trainset_path, "wb") as train_output:
            joblib.dump(df_train, train_output)
        logger.info("Saved trainset at {0}".format(self.trainset_path))

        df_test["relation_id"] = df_test.progress_apply(
            lambda x: rm.rel2idx[x["relation"]], axis=1
        )
        df_test.drop(columns=["relation"], inplace=True)
        with open(self.testset_path, "wb") as test_output:
            joblib.dump(df_test, test_output)
        logger.info("Saved testset at {0}".format(self.testset_path))

        return df_train, df_test, rm

    def tokenize_dataset(self, e1_e2_start, s):
        s = (
            [self.tokenizer.cls_token]
            + self.tokenizer.tokenize(s)
            + [self.tokenizer.sep_token]
        )
        e1_s = s.index("[E1]")
        e2_s = s.index("[E2]")
        e1_e2_start.append([e1_s, e2_s])
        s = self.tokenizer.convert_tokens_to_ids(s)
        return s

    def _preprocess_string(self, text):
        sents, relations, comments, blanks = [], [], [], []
        for i in range(int(len(text) / 4)):
            sent = text[4 * i]
            relation = text[4 * i + 1]
            sent = re.findall('"(.+)"', sent)[0]
            sent = re.sub("<e1>", "[E1]", sent)
            sent = re.sub("</e1>", "[/E1]", sent)
            sent = re.sub("<e2>", "[E2]", sent)
            sent = re.sub("</e2>", "[/E2]", sent)
            sent = self.normalizer.normalize(sent)
            sent = re.sub("\[e1]", "[E1]", sent)
            sent = re.sub("\[/e1]", "[/E1]", sent)
            sent = re.sub("\[e2]", "[E2]", sent)
            sent = re.sub("\[/e2]", "[/E2]", sent)
            sents.append(sent)
            relations.append(relation)
        return sents, relations

    def load_dataset(self):
        if os.path.isfile(self.trainset_path) and os.path.isfile(
            self.testset_path
        ):
            df_train = joblib.load(self.trainset_path)
            df_test = joblib.load(self.testset_path)
            logger.info("Loaded preproccessed data.")
        else:
            df_train, df_test, rm = self.preprocess()

        self.n_classes = len(df_train["relation_id"].unique())
        return df_train, df_test


class RelationsMap(object):
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
