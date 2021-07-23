# coding=utf-8
import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class Normalizer(object):
    def __init__(
        self,
        language="de",
        method_list=(
            "lowercase",
            "html",
            "urls",
        ),
    ):
        """
        Takes a string and returns a normalized string.

        To use it do: Normalizer().normalize('some text to normalize')

        Args:
            language: Expected language for the normalization
            method_list: Which normalization methods to apply
        """
        self.language = language
        self.method_list = method_list
        self.method_map = {
            "lowercase": Normalizer._make_lowercase,
            "html": Normalizer._remove_html,
            "urls": Normalizer._remove_urls,
        }
        self.stop_word_map = {
            "en": stopwords.words("english"),
        }

    def normalize(self, text: str) -> str:
        """
        Normalize the given string.

        Args:
            text: String to normalize

        Returns:
            The normalized string
        """
        text = str(text)
        for method_name in self.method_list:
            text = self.method_map[method_name](text)
        text = re.sub(" +", " ", text)  # remove multiple spaces
        return text.strip()  # strip spaces

    @staticmethod
    def _make_lowercase(text: str) -> str:
        text = text.lower()
        return text

    @staticmethod
    def _remove_html(text: str) -> str:
        return re.sub(re.compile("<.*?>"), "", text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        # Removes URLs of any format, examples: http://www.test.com, www.test.com
        return re.sub(
            r"((https?:\/\/|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",  # noqa: P103
            "",
            text,
        )
