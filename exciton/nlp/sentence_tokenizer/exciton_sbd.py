import json
import os
import re
from typing import Any, Dict, List

import pysbd
from sentence_splitter import SentenceSplitter

from exciton.nlp.lang_detection import Exciton_Lang_Detection


class Exciton_SBD(object):
    """Customized Sentence Boundary Disambiguation.

    Note:
        This is not a standard SBD. The goal is to split text into small pieces.

    Args:
        lang (str): language.
        path_to_model (str, optional): path to model. Defaults to None.
    """

    def __init__(self, lang: str, path_to_model: str = None) -> None:
        if path_to_model is None:
            HOME = os.path.expanduser("~")
            MODEL_DIR = "exciton/models/nlp/sentence_tokenizer/exciton_sbd"
            path_to_model = f"{HOME}/{MODEL_DIR}"
        with open(f"{path_to_model}/languages.json") as fp:
            languages = json.load(fp)
        model = {}
        for key in languages:
            if languages[key]["sbd"] == "exciton":
                model[languages[key]["code_sbd"]] = languages[key]["eos"]
        self.model = re.compile("|".join(model[lang]))

    def split(self, text: str) -> List[str]:
        """Split Text.

        Args:
            text (str): Source text.

        Returns:
            List[str]: List of sentences.
        """
        sents = []
        last_pos = 0
        for itm in self.model.finditer(text):
            if itm.span()[1] < len(text):
                wd = text[itm.span()[1]]
                if wd != " ":
                    continue
            sen = text[last_pos : itm.span()[1]].strip()
            sents.append(sen)
            last_pos = itm.span()[1]
        if last_pos < len(text):
            sen = text[last_pos:].strip()
            sents.append(sen)
        return sents


class Sentence_Tokenizer(object):
    """Sentence Tokenizer.

    Args:
        path_to_model (str, optional): _description_. Defaults to None.

    Note:
        This is not a standard SBD. The goal is to split text into small pieces.
    """

    def __init__(self, path_to_model: str = None) -> None:
        if path_to_model is None:
            HOME = os.path.expanduser("~")
            MODEL_DIR = "exciton/models/nlp/sentence_tokenizer/exciton_sbd"
            path_to_model = f"{HOME}/{MODEL_DIR}"
        with open(f"{path_to_model}/languages.json") as fp:
            languages = json.load(fp)
            self.languages = languages
        self.worker = {}
        # pysbd
        lang_pysbd = [
            languages[key]["code_sbd"]
            for key in languages
            if languages[key]["sbd"] == "pysbd"
        ]
        for lang in lang_pysbd:
            self.worker[lang] = pysbd.Segmenter(language=lang, clean=False).segment
        # sentsplit
        lang_ss = [
            languages[key]["code_sbd"]
            for key in languages
            if languages[key]["sbd"] == "sentsplit"
        ]
        for lang in lang_ss:
            self.worker[lang] = SentenceSplitter(language=lang).split
        # exciton
        lang_exciton = [
            languages[key]["code_sbd"]
            for key in languages
            if languages[key]["sbd"] == "exciton"
        ]
        for lang in lang_exciton:
            self.worker[lang] = Exciton_SBD(
                lang=lang, path_to_model=path_to_model
            ).split
        self.lang_map = {key: self.languages[key]["code_sbd"] for key in self.languages}
        self.lang_sbd = list(
            set([self.languages[key]["code_sbd"] for key in self.languages])
        )
        self.model_ld = Exciton_Lang_Detection(path_to_model=path_to_model)

    def get_support_languages(self) -> List[Dict[str, Any]]:
        """Get support languages.

        Returns:
            List[Dict[str, Any]]: List of languages.
        """
        langs = [
            {
                "code": self.languages[key]["code2"],
                "name": self.languages[key]["name2"],
                "slen": self.languages[key]["sbd_slen"],
            }
            for key in self.languages
        ]
        return langs

    def predict(self, source: str, source_lang: str = None) -> List[str]:
        """Begin to work.

        Args:
            source (str): Source text.
            source_lang (str): source language.

        Returns:
            List[str]: List of sentences.
        """
        if source_lang in self.lang_sbd:
            lang_tgt = source_lang
        elif source_lang in self.languages:
            lang_tgt = self.lang_map[source_lang]
        else:
            source_lang = self.model_ld.predict(source)["code"]
            lang_tgt = self.lang_map[source_lang]
        output = []
        for seg in source.split("\n"):
            out = []
            for sen in self.worker[lang_tgt](seg):
                if len(out) > 0 and len(out[-1]) < 10:
                    out[-1] += " " + sen.strip()
                else:
                    out.append(sen.strip())
            output.extend(out)
        return output
