import json
import os
from multiprocessing import Pool
from typing import Any, Dict, List, Union

from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Lang_Detection(object):
    """Language Detection module implemented by ExcitonX team.

    Args:
        path_to_model (str, optional): path to model. Defaults to None.
    """

    def __init__(self, path_to_model: str = None) -> None:
        if path_to_model is None:
            HOME = os.path.expanduser("~")
            MODEL_DIR = (
                "exciton/models/nlp/lang_detection/exciton_detection/lang_detect100"
            )
            path_to_model = f"{HOME}/{MODEL_DIR}"
        self.tokenizer = AutoTokenizer.from_pretrained(f"{path_to_model}/tokenizer")
        with open(f"{path_to_model}/model.json", "r") as fp:
            self.model_ld = json.load(fp)
        with open(f"{path_to_model}/languages.json", "r") as fp:
            self.langs = json.load(fp)

    def get_support_languages(self) -> List[Dict[str, Any]]:
        """Get support languages.

        Returns:
            List[Dict[str, Any]]: language list.
        """
        output = []
        for key in self.langs:
            output.append(
                {"code": self.langs[key]["code2"], "name": self.langs[key]["name2"]}
            )
        return output

    def predict(self, text: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Precdict Languages.

        Args:
            text (str): Input text

        Returns:
            Dict[str, Any]: Scores.
        """
        if isinstance(text, Dict):
            text = text["text"]
        tokens = self.tokenizer.tokenize(text)
        tokens = list(set(tokens))[:100]
        lang_cand = {lang: 0 for lang in self.model_ld}
        for k, wd in enumerate(tokens):
            for lang in lang_cand:
                if wd in self.model_ld[lang]:
                    lang_cand[lang] += 1
            if k > 8:
                for pct in [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
                    arr = {
                        lang: lang_cand[lang]
                        for lang in lang_cand
                        if lang_cand[lang] / k >= pct
                    }
                    if len(arr) > 0:
                        lang_cand = arr
                        break
        lang_cand = [[wd, lang_cand[wd]] for wd in lang_cand]
        lang_cand = sorted(lang_cand, key=lambda x: x[1])[::-1]
        out = {
            "code": lang_cand[0][0],
            "name": self.langs[lang_cand[0][0]]["name2"],
        }
        return out

    def predict_many(
        self, data: List[Union[Dict[str, Any], str]], n_cpus: int = 8
    ) -> List[Dict[str, Any]]:
        """Predict A batch of text.

        Args:
            data (List[Dict[str, Any]]): Input data.

        Returns:
            List[Dict[str, Any]]: Output.
        """
        pool = Pool(n_cpus)
        return pool.map(self.predict, data)
