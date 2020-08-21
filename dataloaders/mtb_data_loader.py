import itertools
import logging
import os
import re

import joblib
import numpy as np
import pandas as pd
import spacy
import torch
from ml_utils.common import valncreate_dir
from ml_utils.normalizer import Normalizer
from pandarallel import pandarallel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AlbertTokenizer, BertTokenizer

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from dataloaders.mtb_data_generator import MTBGenerator
from src.misc import get_subject_objects

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")

pandarallel.initialize(nb_workers=8, use_memory_fs=False)


class MTBPretrainDataLoader:
    def __init__(self, config: dict):
        """
        DataLoader for MTB data.

        Args:
            config: configuration parameters
        """
        self.config = config
        transformer = self.config.get("transformer")
        self.experiment_name = self.config.get("experiment_name")
        self.tokenizer = MTBPretrainDataLoader.load_tokenizer(transformer)

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.normalizer = Normalizer("en", config.get("normalization", []))
        self.data = self.load_dataset()
        self.train_generator = MTBGenerator(
            data=self.data.copy(),
            batch_size=self.config.get("batch_size"),
            tokenizer=self.tokenizer,
            dataset="train",
        )
        self.validation_generator = MTBGenerator(
            data=self.data.copy(),
            batch_size=self.config.get("batch_size"),
            tokenizer=self.tokenizer,
            dataset="validation",
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
        e1_id = tokenizer.convert_tokens_to_ids("[E1]")
        e2_id = tokenizer.convert_tokens_to_ids("[E2]")
        if e1_id == e2_id:
            raise ValueError("E1 token equals E2 token")
        return tokenizer

    def load_dataset(self):
        """
        Load the data defined in the configuration parameters.
        """
        data_path = self.config.get("data")
        data_file = os.path.basename(data_path)
        data_file_name = os.path.splitext(data_file)[0]
        file_name = "_".join([data_file_name, self.config.get("transformer")])
        preprocessed_file = os.path.join(
            "data", self.experiment_name, file_name + ".pkl"
        )
        preprocessed_raw_data = os.path.join(
            "data", data_file_name + "_extracted.pkl"
        )
        if os.path.isfile(preprocessed_file):
            logger.info("Loaded pre-training data from saved file")
            with open(preprocessed_file, "rb") as pkl_file:
                return joblib.load(pkl_file)

        elif os.path.isfile(preprocessed_raw_data):
            with open(preprocessed_raw_data, "rb") as in_file:
                dataset = joblib.load(in_file)
        else:
            logger.info("Loading Spacy NLP")
            nlp = spacy.load("en_core_web_lg")

            logger.info("Loading pre-training data")
            with open(data_path, "r", encoding="utf8") as f:
                text = f.readlines()
            text = pd.Series(text)
            dataset = text.parallel_apply(func=self._extract_entities, nlp=nlp)
            dataset = pd.concat(dataset.values)
            logger.info(
                "Number of relation statements in corpus: {0}".format(
                    len(dataset)
                )
            )
            del nlp  # noqa: WPS420
            dataset.reset_index(inplace=True, drop=True)
            with open(preprocessed_raw_data, "wb") as preprocessed_path:
                joblib.dump(dataset, preprocessed_path)

        valncreate_dir(os.path.join("data", self.experiment_name))
        dataset = self.preprocess(dataset)
        with open(preprocessed_file, "wb") as fully_preprocessed_path:
            joblib.dump(dataset, fully_preprocessed_path)
        logger.info(
            "Saved pre-training corpus to {0}".format(preprocessed_file)
        )
        return dataset

    def _extract_entities(self, t, nlp):
        t = self._process_textlines([t])
        t = self.normalizer.normalize(t)
        doc = nlp(t)
        return pd.DataFrame(
            self.create_pretraining_dataset(doc, window_size=40),
            columns=["r", "e1", "e2"],
        )

    def preprocess(self, data: pd.DataFrame):
        """
        Preprocess the dataset.

        Normalizes the dataset, tokenizes it and add special tokens

        Args:
            data: dataset to preprocess
        """
        logger.info("Clean dataset")
        idx_to_pop = set()
        groups = data.groupby(["e1", "e2"])
        for _idx, group in tqdm(groups, total=len(groups)):
            if len(group) < self.config.get("min_pool_size", 2):
                for i in group.index.values.tolist():
                    idx_to_pop.add(i)
        data = data.drop(index=idx_to_pop).reset_index(drop=True)
        data["relation_id"] = np.arange(0, len(data))

        logger.info("Normalizing relations")
        normalized_relations = []
        for _idx, row in tqdm(data.iterrows(), total=len(data)):
            relation = self._add_special_tokens(row)
            normalized_relations.append(relation)

        logger.info("Tokenizing relations")
        tokenized_relations = []
        for n in tqdm(normalized_relations):
            tokenized_relations.append(
                torch.IntTensor(self.tokenizer.convert_tokens_to_ids(n))
            )
        del normalized_relations  # noqa: WPS 420

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
        del tokenized_relations  # noqa: WPS 420
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
            relation.append(w)
            if w_idx == e_span1[1]:
                relation.append("[/E1]")
            if w_idx == e_span2[1]:
                relation.append("[/E2]")
        relation.append(self.tokenizer.sep_token)
        return relation

    def transform_data(self, df: pd.DataFrame):
        """
        Prepare data for the QQModel.

        Data format:     Question pairs1.     Question pairs2. Negative
        question pool per question.

        Args:
            df: Dataframe to use to generate QQ pairs.
        """
        pools = self.generate_entities_pools(df)
        for idx, pool in enumerate(pools):
            if np.random.random() > 0.75:
                pools[idx] = pool + ("validation",)
            else:
                pools[idx] = pool + ("train",)
        return pools

    def generate_entities_pools(self, data: pd.DataFrame):
        """
        Generate class pools.

        Args:
            data: pandas dataframe containing the relation, entity 1 & 2 and the relation id

        Returns:
            Index of question.
            Index of paired question.
            Common answer id.
        """
        logger.info("Generating class pools")
        pool = []
        groups = data.groupby(["e1", "e2"])
        groups_e1 = data.groupby(["e1"])
        groups_e2 = data.groupby(["e2"])
        for idx, group in tqdm(groups, total=len(groups)):
            e1, e2 = idx
            data_e1 = groups_e1.get_group(e1)
            data_e2 = groups_e2.get_group(e2)
            e1_negatives = data_e1.loc[data_e1["e2"] != e2, "relation_id"]
            e2_negatives = data_e2.loc[data_e2["e1"] != e1, "relation_id"]
            entities_pool = (
                group["relation_id"].values.tolist(),
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
            doc: spacy doc
            window_size: Maximum windows size between to entities
        """
        ents = doc.ents  # get entities

        entities_of_interest = self.config.get("entities_of_interest")
        length_doc = len(doc)
        data = []
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
            if e1.text.lower() == e2.text.lower():  # make sure e1 != e2
                continue
            # check if next nearest entity within window_size
            if 1 <= (e2start - e1end) <= window_size:
                # Find start of sentence
                left_r = MTBPretrainDataLoader._find_end_of_sentence(
                    doc, e1start
                )

                # Find end of sentence
                right_r = MTBPretrainDataLoader._find_start_of_sentence(
                    doc, e2end, length_doc
                )

                # sentence should not be longer than window_size
                if (right_r - left_r) > window_size:
                    continue

                x = [token.text for token in doc[left_r:right_r]]

                empty_token = all(not token for token in x)
                emtpy_e1 = not e1.text
                emtpy_e2 = not e2.text
                emtpy_span = (e2start - e1end) < 1
                if emtpy_e1 or emtpy_e2 or emtpy_span or empty_token:
                    raise ValueError("Relation has wrong format")

                r = (
                    x,
                    (e1start - left_r, e1end - left_r),
                    (e2start - left_r, e2end - left_r),
                )
                data.append((r, e1.text, e2.text))
                ents_list.append((e1.text, e2.text))

        doc_sents = list(doc.sents)
        for sent in doc_sents:
            if len(sent) > (window_size + 1):
                continue

            left_r = sent[0].i
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
                    ent1[-1].i + 1 if isinstance(ent1, list) else ent1.i + 1,
                )
                e2start, e2end = (
                    ent2[0].i if isinstance(ent2, list) else ent2.i,
                    ent2[-1].i + 1 if isinstance(ent2, list) else ent2.i + 1,
                )
                if (e1end < e2start) and ((e1text, e2text) not in ents_list):
                    if (e2start - e1end) <= 0:
                        raise ValueError("e2start is smaller than e1end")
                    r = (
                        [w.text for w in sent],
                        (e1start - left_r, e1end - left_r),
                        (e2start - left_r, e2end - left_r),
                    )
                    data.append((r, e1text, e2text))
                    ents_list.append((e1text, e2text))
        return data

    @classmethod
    def _find_end_of_sentence(cls, doc, e1start):
        punc_token = False
        start = e1start - 1
        if start > 0:
            while not punc_token:
                punc_token = doc[start].is_punct
                start -= 1
                if start < 0:
                    break
            left_r = start + 2 if start > 0 else 0
        else:
            left_r = 0
        return left_r

    @classmethod
    def _find_start_of_sentence(cls, doc, e2end, length_doc):
        punc_token = False
        start = e2end
        if start < length_doc:
            while not punc_token:
                punc_token = doc[start].is_punct
                start += 1
                if start == length_doc:
                    break
            right_r = start if start < length_doc else length_doc
        else:
            right_r = length_doc
        return right_r
