import gc
import itertools
import logging
import os
import re

from tqdm import tqdm

import joblib
import numpy as np
import pandas as pd
import spacy
import torch
from dataloaders.mtb_data_generator import MTBGenerator
from ml_utils.common import valncreate_dir
from ml_utils.normalizer import Normalizer
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")


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
        preprocessed_folder = os.path.join("data", self.experiment_name)
        valncreate_dir(preprocessed_folder)
        preprocessed_file = os.path.join(
            preprocessed_folder, file_name + ".pkl"
        )
        build_dataset_file = os.path.join(
            preprocessed_folder, data_file_name + "_extracted.pkl"
        )

        if os.path.isfile(preprocessed_file):
            with open(preprocessed_file, "rb") as pkl_file:
                logger.info("Loaded pre-training data from saved file")
                return joblib.load(pkl_file)

        elif os.path.isfile(build_dataset_file):
            with open(build_dataset_file, "rb") as in_file:
                dataset = joblib.load(in_file)
        else:
            logger.info("Building pretraining dataset from corpus")
            with open(data_path, "r", encoding="utf8") as f:
                text = f.readlines()

            dataset = self.build_dataset(text, build_dataset_file)

        valncreate_dir(os.path.join("data", self.experiment_name))
        dataset = self.preprocess(dataset)
        with open(preprocessed_file, "wb") as fully_preprocessed_path:
            joblib.dump(dataset, fully_preprocessed_path)
        logger.info(
            "Saved pre-training corpus to {0}".format(preprocessed_file)
        )
        return dataset

    def build_dataset(self, text, save_path):
        """
        Builds the Matching the Blanks pretraining dataset from the given
        textcorpus.

        Args:
            text: List of text corpora
            save_path: Where to save the file
        """
        dataset, x_map_rev, e_map_rev = self._build_mapped_dataset(text)
        logger.info(
            "Number of relation statements in corpus: {0}".format(len(dataset))
        )
        for idx, r in tqdm(dataset.iterrows(), total=len(dataset)):
            x = r.get("r")[0]
            e1 = r.get("e1")
            e2 = r.get("e2")
            sent = x_map_rev[x]
            dataset["r"][idx] = (
                sent,
                dataset["r"][idx][1],
                dataset["r"][idx][2],
            )
            dataset["e1"][idx] = e_map_rev[e1]
            dataset["e2"][idx] = e_map_rev[e2]
        with open(save_path, "wb") as preprocessed_path:
            joblib.dump(dataset, preprocessed_path)
        return dataset

    def _build_mapped_dataset(self, text):
        logger.info("Loading Spacy NLP")
        nlp = spacy.load("en_core_web_lg")
        dataset = []
        x_map = {}
        x_map_rev = {}
        e_map = {}
        e_map_rev = {}
        x_idx = 0
        e_idx = 0

        for idx_t, t in enumerate(tqdm(text)):
            dataset_t = self._extract_entities(t, nlp)
            for idx, r in dataset_t.iterrows():
                x = r.get("r")
                e1 = r.get("e1")
                e2 = r.get("e2")
                x = x[0]
                x_join = " ".join(x)
                if x_join not in x_map:
                    x_map[x_join] = x_idx
                    x_map_rev[x_idx] = x
                    dataset_t["r"][idx] = (
                        x_idx,
                        dataset_t["r"][idx][1],
                        dataset_t["r"][idx][2],
                    )
                    x_idx += 1
                else:
                    k = x_map[x_join]
                    dataset_t["r"][idx] = (
                        k,
                        dataset_t["r"][idx][1],
                        dataset_t["r"][idx][2],
                    )
                if e1 not in e_map:
                    e_map[e1] = e_idx
                    e_map_rev[e_idx] = e1
                    dataset_t["e1"][idx] = e_idx
                    e_idx += 1

                else:
                    dataset_t["e1"][idx] = e_map[e1]

                if e2 not in e_map:
                    e_map[e2] = e_idx
                    e_map_rev[e_idx] = e2
                    dataset_t["e2"][idx] = e_idx
                    e_idx += 1

                else:
                    dataset_t["e2"][idx] = e_map[e2]

                if idx_t % 1000 == 0:
                    gc.collect()

            dataset.append(dataset_t)
            text[idx_t] = None
        dataset = pd.concat(dataset)
        dataset.reset_index(inplace=True, drop=True)
        return dataset, x_map_rev, e_map_rev

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
        for _group_id, group in tqdm(groups, total=len(groups)):
            if len(group) < self.config.get("min_pool_size", 2):
                for i in group.index.values.tolist():
                    idx_to_pop.add(i)
        data = data.drop(index=idx_to_pop).reset_index(drop=True)
        data["relation_id"] = np.arange(0, len(data))

        logger.info("Normalizing relations")
        normalized_relations = []
        for _row_id, row in tqdm(data.iterrows(), total=len(data)):
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
        length_doc = len(doc)
        data = []
        ents_list = []

        spans = list(doc.ents) + list(doc.noun_chunks)
        spans = spacy.util.filter_spans(spans)
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
        doc_ents = doc.ents
        ents_text = set()
        ents = []
        for e in itertools.chain(spans, doc_ents):
            if e.text not in ents_text:
                ents.append(e)
                ents_text.add(e.text)

        for e1, e2 in itertools.product(ents, ents):
            if e1 == e2:
                continue
            e1start = e1.start
            e1end = e1.end - 1
            e2start = e2.start
            e2end = e2.end - 1
            e1_has_numbers = re.search("[\d+]", e1.text)
            e2_has_numbers = re.search("[\d+]", e2.text)
            if e1_has_numbers or e2_has_numbers:
                continue
            if 1 <= (e2start - e1end) <= window_size:
                # Find start of sentence
                r_start = MTBPretrainDataLoader._find_sent_start(doc, e1start)
                r_end = MTBPretrainDataLoader._find_sent_end(
                    doc, e2end, length_doc
                )

                # sentence should not be longer than window_size
                if (r_end - r_start) > window_size:
                    continue

                x = [token.text for token in doc[r_start:r_end]]

                empty_token = all(not token for token in x)
                emtpy_e1 = not e1.text
                emtpy_e2 = not e2.text
                emtpy_span = (e2start - e1end) < 1
                if emtpy_e1 or emtpy_e2 or emtpy_span or empty_token:
                    raise ValueError("Relation has wrong format")

                r = (
                    x,
                    (e1start - r_start, e1end - r_start),
                    (e2start - r_start, e2end - r_start),
                )
                data.append((r, e1.text, e2.text))
                ents_list.append((e1.text, e2.text))
        return data

    @classmethod
    def _find_sent_start(cls, doc, e1start):
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
    def _find_sent_end(cls, doc, e2end, length_doc):
        sent_start = False
        start = e2end
        if start < length_doc:
            while not sent_start:
                sent_start = doc[start].is_sent_end
                sent_start = sent_start if sent_start else False
                start += 1
                if start == length_doc:
                    break
            right_r = start if start < length_doc else length_doc
        else:
            right_r = length_doc
        return right_r
