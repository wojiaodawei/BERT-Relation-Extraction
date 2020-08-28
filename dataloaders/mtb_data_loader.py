import itertools
import json
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
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from dataloaders.mtb_data_generator import MTBGenerator

tqdm.pandas()

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
        file_name = "_".join(
            [
                data_file_name,
                self.config.get("transformer"),
                str(self.config.get("min_pool_size")),
            ]
        )
        preprocessed_folder = os.path.join("data", self.experiment_name)
        valncreate_dir(preprocessed_folder)
        preprocessed_file = os.path.join(
            preprocessed_folder, file_name + ".pkl"
        )

        if os.path.isfile(preprocessed_file):
            with open(preprocessed_file, "rb") as pkl_file:
                logger.info("Loaded pre-training data from saved file")
                return joblib.load(pkl_file)

        else:
            logger.info("Building pretraining dataset from corpus")
            with open(data_path, "r", encoding="utf8") as f:
                text = f.readlines()

            dataset, x_map_rev, e_map_rev = self.build_dataset(
                text, preprocessed_file
            )

        valncreate_dir(os.path.join("data", self.experiment_name))
        dataset = self.preprocess(dataset, x_map_rev, e_map_rev)
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
        save_dir = os.path.dirname(save_path)
        x_map_rev_path = os.path.join(save_dir, "x_map_rev.json")
        e_map_rev_path = os.path.join(save_dir, "e_map_rev.json")
        mapped_dataset_path = os.path.join(save_dir, "mapped.npy")
        if (
            os.path.isfile(x_map_rev_path)
            and os.path.isfile(e_map_rev_path)
            and os.path.isfile(mapped_dataset_path)
        ):
            with open(x_map_rev_path) as x_f:
                x_map_rev = json.load(x_f)
                x_map_rev = {int(k): v for k, v in x_map_rev.items()}
            with open(e_map_rev_path) as x_f:
                e_map_rev = json.load(x_f)
                e_map_rev = {int(k): v for k, v in e_map_rev.items()}
            dataset = np.load(mapped_dataset_path)
        else:
            dataset, x_map_rev, e_map_rev = self._build_mapped_dataset(
                text, save_dir
            )
        logger.info(
            "Number of relation statements in corpus: {0}".format(len(dataset))
        )
        return dataset, x_map_rev, e_map_rev

    @classmethod
    def _build_dataframe(cls, r, x_map, e_map):
        x = r.get("x")
        e1 = r.get("e1")
        e2 = r.get("e2")
        r["r"] = (
            x_map[x],
            (r.get("e1_start"), r.get("e1_end")),
            (r.get("e2_start"), r.get("e2_end")),
        )
        r["e1"] = e_map[e1]
        r["e2"] = e_map[e2]
        return r

    def _build_mapped_dataset(self, text, save_dir):
        dataset, x_map_rev, e_map_rev = self._extract_entities(text)
        np.save(os.path.join(save_dir, "mapped.npy"), dataset)
        with open(
            os.path.join(save_dir, "x_map_rev.json"), "w", encoding="utf-8"
        ) as x_f:
            json.dump(x_map_rev, x_f, ensure_ascii=False, indent=4)
        with open(
            os.path.join(save_dir, "e_map_rev.json"), "w", encoding="utf-8"
        ) as e_f:
            json.dump(e_map_rev, e_f, ensure_ascii=False, indent=4)
        return dataset, x_map_rev, e_map_rev

    def _extract_entities(self, texts):
        texts = [self._process_textlines([t]) for t in texts]
        texts = [self.normalizer.normalize(t) for t in texts]
        n_texts = len(texts)
        logger.info("Loading Spacy NLP")
        nlp = spacy.load("en_core_web_lg")
        docs = nlp.pipe(texts)
        return self.create_pretraining_dataset(docs, n_texts, window_size=40)

    def preprocess(self, data: np.ndarray, x_map_rev: dict, e_map_rev: dict):
        """
        Preprocess the dataset.

        Normalizes the dataset, tokenizes it and add special tokens

        Args:
            data: dataset to preprocess
        """
        ee_map_freq = {}
        logger.info("Build entity frequency map")
        for e1, e2 in tqdm(data[:, [5, 6]]):
            tuple_key = str(e1) + "_" + str(e2)
            if tuple_key not in ee_map_freq:
                ee_map_freq[tuple_key] = 1
            else:
                ee_map_freq[tuple_key] += 1

        min_pool_size = self.config.get("min_pool_size")
        if min_pool_size > 2:
            logger.info(
                f"Remove Entity combinations with less than "
                f"{min_pool_size} representations"
            )
            (
                data,
                x_map_rev,
                e_map_rev,
            ) = MTBPretrainDataLoader.remove_low_freq_combs(
                data, ee_map_freq, x_map_rev, e_map_rev, min_pool_size
            )
        data = pd.DataFrame(
            data,
            columns=[
                "x",
                "e1_start",
                "e1_end",
                "e2_start",
                "e2_end",
                "e1",
                "e2",
            ],
        )

        data = data.progress_apply(
            MTBPretrainDataLoader._build_dataframe,
            e_map=e_map_rev,
            x_map=x_map_rev,
            axis=1,
        )
        data.drop(
            columns=["x", "e1_start", "e1_end", "e2_start", "e2_end"],
            inplace=True,
        )
        data["relation_id"] = np.arange(0, len(data))

        logger.info("Add special tokens to relations")
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

    def create_pretraining_dataset(self, docs, n_docs, window_size: int = 40):
        """
        Input: Chunk of raw text
        Output: modified corpus of triplets (relation statement, entity1, entity2)

        Args:
            docs: Iterable spacy doc
            window_size: Maximum windows size between to entities
            n_docs: Number of docs
        """
        ovr_dataset = []
        x_map = {}
        x_map_rev = {}
        e_map = {}
        e_map_rev = {}
        ee_map_freq = {}
        x_idx = 0
        e_idx = 0

        for doc in tqdm(docs, total=n_docs):
            data = []
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

            ee_product = list(itertools.product(ents, ents))
            for e in ee_product:
                data.append(
                    self._resolve_entities(e, doc=doc, window_size=window_size)
                )
            data = [d for d in data if d]
            for idx, d in enumerate(data):
                x = d[0]
                e1 = d[5]
                e2 = d[6]
                x_join = " ".join(x)
                if x_join not in x_map:
                    x_map[x_join] = x_idx
                    x_map_rev[x_idx] = x
                    new_r = [x_idx, d[1], d[2], d[3], d[4]]
                    x_idx += 1
                else:
                    k = x_map[x_join]
                    new_r = [k, d[1], d[2], d[3], d[4]]
                if e1 not in e_map:
                    e_map[e1] = e_idx
                    e_map_rev[e_idx] = e1
                    new_e1 = e_idx
                    e_idx += 1

                else:
                    new_e1 = e_map[e1]

                if e2 not in e_map:
                    e_map[e2] = e_idx
                    e_map_rev[e_idx] = e2
                    new_e2 = e_idx
                    e_idx += 1
                else:
                    new_e2 = e_map[e2]
                tuple_key = str(new_e1) + "_" + str(new_e2)
                if tuple_key not in ee_map_freq:
                    ee_map_freq[tuple_key] = 1
                else:
                    ee_map_freq[tuple_key] += 1
                data[idx] = new_r + [new_e1] + [new_e2]
            ovr_dataset.extend(data)

        (
            ovr_dataset,
            x_map_rev,
            e_map_rev,
        ) = MTBPretrainDataLoader.remove_low_freq_combs(
            ovr_dataset, ee_map_freq, x_map_rev, e_map_rev
        )

        return ovr_dataset, x_map_rev, e_map_rev

    @classmethod
    def remove_low_freq_combs(
        cls, data, ee_map_freq, x_map_rev, e_map_rev, min_count: int = 2
    ):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        logger.info("Remove underrepresented entity combinations")
        to_idx = 0
        for d in tqdm(data):
            tuple_key = str(d[5]) + "_" + str(d[6])
            if ee_map_freq[tuple_key] >= min_count:
                data[to_idx] = d
                to_idx += 1
        del data[to_idx:]

        data = np.array(data)

        logger.info("Clean Entity Map")
        unique_entities = set(np.unique(data[:, [5, 6]].reshape(1, -1)[0]))
        idx_to_pop = set()
        for e_key in tqdm(e_map_rev.keys()):
            if e_key not in unique_entities:
                idx_to_pop.add(e_key)

        for idx in idx_to_pop:
            e_map_rev.pop(idx)

        logger.info("Clean X Map")
        unique_x = set(np.unique(data[:, 0]))
        idx_to_pop = set()
        for x_key in tqdm(x_map_rev.keys()):
            if x_key not in unique_x:
                idx_to_pop.add(x_key)

        for idx in idx_to_pop:
            x_map_rev.pop(idx)

        return data, x_map_rev, e_map_rev

    def _resolve_entities(self, e, doc, window_size):
        e1, e2 = e
        if e1 == e2:
            return None
        e1start = e1.start
        e1end = e1.end - 1
        e2start = e2.start
        e2end = e2.end - 1
        e1_has_numbers = re.search("[\d+]", e1.text)
        e2_has_numbers = re.search("[\d+]", e2.text)
        if e1_has_numbers or e2_has_numbers:
            return None
        if 1 <= (e2start - e1end) <= window_size:
            length_doc = len(doc)
            # Find start of sentence
            r_start = MTBPretrainDataLoader._find_sent_start(doc, e1start)
            r_end = MTBPretrainDataLoader._find_sent_end(
                doc, e2end, length_doc
            )

            # sentence should not be longer than window_size
            if (r_end - r_start) > window_size:
                return None

            x = [token.text for token in doc[r_start:r_end]]

            empty_token = all(not token for token in x)
            emtpy_e1 = not e1.text
            emtpy_e2 = not e2.text
            emtpy_span = (e2start - e1end) < 1
            if emtpy_e1 or emtpy_e2 or emtpy_span or empty_token:
                raise ValueError("Relation has wrong format")

            r = [
                x,
                e1start - r_start,
                e1end - r_start,
                e2start - r_start,
                e2end - r_start,
            ]
            return r + [e1.text] + [e2.text]

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
