import itertools
import logging
import os
import re

import joblib
import neuralcoref
import numpy as np
import pandas as pd
import spacy
import torch
from ml_utils.normalizer import Normalizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AlbertTokenizer

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from dataloaders.mtb_data_generator import MTBTrainGenerator
from src.misc import get_subject_objects

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")


class DataLoader:
    def __init__(self, config: dict):
        """
        DataLoader for MTB data.

        Args:
            config: configuration parameters
        """
        self.config = config

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
        if e1_id == e2_id:
            raise ValueError("E1 token equals E2 token")

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.normalizer = Normalizer("en", config.get("normalization", []))
        self.data = self.load_dataset()
        self.train_generator = MTBTrainGenerator(
            dataset=self.data,
            batch_size=self.config.get("batch_size"),
            tokenizer=self.tokenizer,
        )

    def load_dataset(self):
        """
        Load the data defined in the configuration parameters.
        """
        data_path = self.config.get("data")
        data_file = os.path.basename(data_path)
        data_file_name = os.path.splitext(data_file)[0]
        preprocessed_file = os.path.join("data", data_file_name + ".pkl")

        if os.path.isfile(preprocessed_file):
            logger.info("Loaded pre-training data from saved file")
            with open(preprocessed_file, "rb") as pkl_file:
                data = joblib.load(pkl_file)

        else:
            logger.info("Loading pre-training data...")
            with open(data_path, "r", encoding="utf8") as f:
                text = f.readlines()

            text = self._process_textlines(text)

            logger.info("Length of text (characters): {0}".format(len(text)))

            dataset = []
            logger.info("Loading Spacy NLP")
            nlp = spacy.load("en_core_web_lg")
            neuralcoref.add_to_pipe(nlp)  # Add neural co-reference resolution
            doc = nlp(text)
            doc = doc._.coref_resolved
            sents = list(nlp(doc).sents)

            for sent in tqdm(sents):
                sent_doc = nlp(" ".join([w.text for w in sent]))
                dataset.extend(
                    self.create_pretraining_dataset(sent_doc, window_size=40)
                )

            logger.info(
                "Number of relation statements in corpus: {0}".format(
                    len(dataset)
                )
            )
            dataset = pd.DataFrame(dataset)
            dataset.columns = ["r", "e1", "e2"]

            data = self.preprocess(dataset)
            with open(preprocessed_file, "wb") as output:
                joblib.dump(data, output)
            logger.info(
                "Saved pre-training corpus to {0}".format(preprocessed_file)
            )
        return data

    def preprocess(self, data: pd.DataFrame):
        """
        Preprocess the dataset.

        Normalizes the dataset, tokenizes it and add special tokens

        Args:
            data: dataset to preprocess
        """
        logger.info("Normalizing relations")
        normalized_relations = []
        for _idx, row in data.iterrows():
            relation = self._add_special_tokens(row)
            normalized_relations.append(relation)

        logger.info("Tokenizing relations")
        tokenized_relations = [
            torch.IntTensor(self.tokenizer.convert_tokens_to_ids(n))
            for n in normalized_relations
        ]
        tokenized_relations = pad_sequence(
            tokenized_relations,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        e_span1 = [(r[1][0] + 2, r[1][1] + 2) for r in data["r"]]
        e_span2 = [(r[2][0] + 4, r[2][1] + 4) for r in data["r"]]
        r = [
            (tr.numpy().tolist(), e1, e2)
            for tr, e1, e2 in zip(tokenized_relations, e_span1, e_span2)
        ]
        data["r"] = r
        pools = self.transform_data(data)
        preprocessed_data = {
            "entities_pools": pools,
            "tokenized_relations": data,
        }
        return preprocessed_data

    def _add_special_tokens(self, row):
        r = row.get("r")[0]
        e_span1 = row.get("r")[1]
        e_span2 = row.get("r")[2]
        relation = [self.tokenizer.cls_token]
        for w_idx, w in enumerate(r):
            if w_idx == e_span1[0]:
                relation.append("[E1]")
            if w_idx == e_span2[0]:
                relation.append("[E2]")
            relation.append(self.normalizer.normalize(w))
            if w_idx == e_span1[1]:
                relation.append("[/E1]")
            if w_idx == e_span2[1]:
                relation.append("[/E2]")
        relation.append(self.tokenizer.sep_token)
        return relation

    @classmethod
    def transform_data(cls, df: pd.DataFrame):
        """
        Prepare data for the QQModel.

        Data format:     Question pairs1.     Question pairs2. Negative
        question pool per question.

        Args:
            df: Dataframe to use to generate QQ pairs.
        """
        df["relation_id"] = np.arange(0, len(df))
        logger.info("Generating class pools")
        return DataLoader.generate_entities_pools(df)

    @classmethod
    def generate_entities_pools(cls, data: pd.DataFrame):
        """
        Generate class pools.

        Args:
            data: pandas dataframe containing the relation, entity 1 & 2 and the relation id

        Returns:
            Index of question.
            Index of paired question.
            Common answer id.
        """
        groups = data.groupby(["e1", "e2"])
        pool = []
        for idx, df in groups:
            e1, e2 = idx
            e1_negatives = data[((data["e1"] == e1) & (data["e2"] != e2))][
                "relation_id"
            ]
            e2_negatives = data[((data["e1"] != e1) & (data["e2"] == e2))][
                "relation_id"
            ]
            entities_pool = (
                df["relation_id"].values.tolist(),
                e1_negatives.values.tolist(),
                e2_negatives.values.tolist(),
            )
            pool.append(entities_pool)
        logger.info("Found {0} different pools".format(len(pool)))
        return pool

    def _process_textlines(self, text):
        text = [self._clean_sent(sent) for sent in text]
        text = " ".join([t for t in text if t is not None])
        text = re.sub(" {2,}", " ", text)
        return text

    @classmethod
    def _clean_sent(cls, sent):
        if sent not in {" ", "\n", ""}:
            sent = sent.strip("\n")
            sent = re.sub(
                "<[A-Z]+/*>", "", sent
            )  # remove special tokens eg. <FIL/>, <S>
            sent = re.sub(
                r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#]", " ", sent
            )
            sent = " ".join(sent.split())  # remove whitespaces > 1
            sent = sent.strip()
            sent = re.sub(
                r"([\.\?,!]){2,}", r"\1", sent
            )  # remove multiple puncs
            sent = re.sub(
                r"([A-Z]{2,})", lambda x: x.group(1).capitalize(), sent
            )  # Replace all CAPS with capitalize
            return sent

    def create_pretraining_dataset(self, doc, window_size: int = 40):
        """
        Input: Chunk of raw text
        Output: modified corpus of triplets (relation statement, entity1, entity2)

        Args:
            doc: spacy doc containing exactly one sentence
            window_size: Maximum windows size between to entities
        """
        ents = doc.ents  # get entities

        entities_of_interest = self.config.get("entities_of_interest")
        len_doc = len(doc)
        data = []
        if len_doc > window_size:
            return data

        ents_list = []
        for e1, e2 in itertools.product(ents, ents):
            if e1 == e2:
                continue
            e1start = e1.start
            e1end = e1.end - 1
            e2start = e2.start
            e2end = e2.end - 1
            e1_has_numbers = re.search("[\d+]", e1.text)
            e2_has_numbers = re.search("[\d+]", e2.text)
            if (e1.label_ not in entities_of_interest) or e1_has_numbers:
                continue
            if (e2.label_ not in entities_of_interest) or e2_has_numbers:
                continue
            if e1.text.lower() == e2.text.lower():
                continue
            # check if next nearest entity within window_size
            if 1 <= (e2start - e1end) <= window_size:
                x = [token.text for token in doc]

                empty_token = all(not token for token in x)
                emtpy_e1 = not e1.text
                emtpy_e2 = not e2.text
                emtpy_span = (e2start - e1end) < 1
                if emtpy_e1 or emtpy_e2 or emtpy_span or empty_token:
                    raise ValueError("Relation has wrong format")

                r = (
                    x,
                    (e1start, e1end),
                    (e2start, e2end),
                )
                data.append((r, e1.text, e2.text))
                ents_list.append((e1.text, e2.text))

        if len_doc > (window_size + 1):
            return data
        for sent in doc.sents:
            pairs = get_subject_objects(sent)
            for pair in pairs:
                ent1, ent2 = pair[0], pair[1]

                if (len(ent1) > 3) or (len(ent2) > 3):
                    continue

                e1text, e2text = (
                    " ".join(w.text for w in ent1)
                    if isinstance(ent1, list)
                    else ent1.text,
                    " ".join(w.text for w in ent2)
                    if isinstance(ent2, list)
                    else ent2.text,
                )
                e1start, e1end = (
                    ent1[0].i if isinstance(ent1, list) else ent1.i,
                    ent1[-1].i if isinstance(ent1, list) else ent1.i,
                )
                e2start, e2end = (
                    ent2[0].i if isinstance(ent2, list) else ent2.i,
                    ent2[-1].i if isinstance(ent2, list) else ent2.i,
                )
                if (e1end < e2start) and ((e1text, e2text) not in ents_list):
                    if (e2start - e1end) <= 0:
                        raise ValueError("e2start is smaller than e1end")
                    r = (
                        [w.text for w in sent],
                        (e1start, e1end),
                        (e2start, e2end),
                    )
                    data.append((r, e1text, e2text))
                    ents_list.append((e1text, e2text))
        return data
