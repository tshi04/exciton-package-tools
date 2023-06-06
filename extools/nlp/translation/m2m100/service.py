import json
import os
from typing import Any, Dict, List, Literal

import ctranslate2
import transformers


class M2M100(object):
    """Machine Translation Module.

    Args:
        path_to_model (str, optional): path to the models. Defaults to None.
        device (Literal[&quot;cpu&quot;, &quot;cuda&quot;], optional): Device. Defaults to "cpu".
    """

    def __init__(
        self,
        path_to_model: str = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        if not path_to_model:
            HOME = os.path.expanduser("~")
            MODEL_DIR = "exciton/models/nlp/translation/m2m100/m2m100_1.2b"
            path_to_model = f"{HOME}/{MODEL_DIR}"
        self.translator = ctranslate2.Translator(path_to_model, device=device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            f"{path_to_model}/tokenizer/"
        )
        with open(f"{path_to_model}/languages.json") as fp:
            languages = json.load(fp)
        languages = {
            key: languages[key] for key in languages if languages[key]["mt"] == "m2m100"
        }
        self.languages = languages
        self.lang_mm = []
        self.lang_map = {}
        for key in languages:
            self.lang_mm.append(languages[key]["code_mt"])
            self.lang_map[languages[key]["code2"]] = languages[key]["code_mt"]

    def get_support_languages(self) -> List[Dict[str, str]]:
        """Suppport Languages for M2M100

        Returns:
            List[Dict[str, str]]: List of languages supported by M2M100 models.
        """
        return [
            {"code": self.languages[key]["code2"], "name": self.languages[key]["name2"]}
            for key in self.languages
        ]

    def _batch(
        self, source: List[Dict[str, Any]], batch_size: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """Create Batch.

        Args:
            source (List[Any]): Source List
            batch_size (int, optional): Batch Size. Defaults to 10.

        Returns:
            List[List[Any]]: Batches
        """
        output = []
        arr = []
        for k, itm in enumerate(source):
            itm["sent_id"] = k
            arr.append(itm)
            if len(arr) == batch_size:
                output.append(arr)
                arr = []
        if len(arr) > 0:
            output.append(arr)
        return output

    def predict(self, source: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch Translation.

        Args:
            source (List[Dict[str, Any]]): Source, input data.

        Returns:
            List[Dict[str, Any]]: Target, output data.

        Examples:
            >>> from exciton.nlp.translation import M2M100
            >>> model = M2M100(model="m2m100_418m", device="cuda")
            >>> source = [
                    {"source": "I love you!", "source_lang": "en", "target_lang": "zh"},
                    {"source": "我爱你！", "source_lang": "zh", "target_lang": "en"}
                ]
            >>> results = model.predict(source)
            >>> print(results)
        """
        output = []
        for batch in self._batch(source):
            sen_list = []
            target_prefix = []
            for itm in batch:
                slang = itm["source_lang"]
                tlang = itm["target_lang"]
                if slang in self.lang_map and tlang in self.lang_map:
                    if isinstance(itm["source"], str):
                        slang = self.lang_map[slang]
                        tlang = self.lang_map[tlang]
                        self.tokenizer.src_lang = slang
                        sen = self.tokenizer.encode(itm["source"])
                        sen = self.tokenizer.convert_ids_to_tokens(sen)
                        sen_list.append(sen)
                        target_prefix.append([self.tokenizer.lang_code_to_token[tlang]])
                    else:
                        itm["target"] = ""
                        output.append(itm)
                else:
                    itm["target"] = ""
                    output.append(itm)
            results = self.translator.translate_batch(
                sen_list,
                target_prefix=target_prefix,
                beam_size=3,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
            )
            for k, out in enumerate(results):
                itm = batch[k]
                out = out.hypotheses[0][1:]
                out = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(out))
                itm["target"] = out
                output.append(itm)
        output = sorted(output, key=lambda x: x["sent_id"])
        return output
