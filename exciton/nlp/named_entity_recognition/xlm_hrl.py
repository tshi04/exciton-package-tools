import json
import os
from typing import Any, Dict, List

import torch
from torch.autograd import Variable
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .utils import clean_result


class Tagging_Model(object):
    """Tagging Model.

    Args:
        path_to_model (str, optional): path to the models. Defaults to None.
        drop_rate (float, optional): dropout rate. Defaults to 0.1.
        n_classes (int, optional): number of labels. Defaults to 0.
        device (str, optional): device. Defaults to "cpu".
    """

    def __init__(
        self,
        path_to_model: str = None,
        drop_rate: float = 0.1,
        n_classes: int = 0,
        device: str = "cpu",
    ):
        self.path_to_model = path_to_model
        self.drop_rate = drop_rate
        self.n_classes = n_classes
        self.device = torch.device(device)
        self.train_modules = {}
        self.base_modules = {}
        self.batch_data = {}

        if self.path_to_model is None:
            HOME = os.path.expanduser("~")
            MODEL_DIR = "exciton/models/nlp/named_entity_recognition"
            model = "xlm_roberta_large_ner_hrl"
            path_to_model = f"{HOME}/{MODEL_DIR}/{model}"
            self.path_to_model = path_to_model
        self.tokenizer = AutoTokenizer.from_pretrained(f"{path_to_model}/tokenizer")

    def build_modules(self):
        """Declare all modules in your model."""
        self.base_modules["model"] = AutoModelForTokenClassification.from_pretrained(
            f"{self.path_to_model}/models",
            output_hidden_states=True,
            output_attentions=True,
        ).to(self.device)
        self.TOK_START = 0
        self.TOK_END = 2
        self.TOK_PAD = 1

    def _build_pipe(self):
        """Shared pipe"""
        with torch.no_grad():
            logits = self.base_modules["model"](
                self.batch_data["input_ids"],
                self.batch_data["pad_mask"],
            )
        return logits[0]

    def _build_batch(self, batch_raw: List[Any]):
        """Build Batch

        Args:
            batch_data (List[Any]): batch data.
        """
        tokens_arr = []
        input_data_raw = []
        max_length = 0
        for itm in batch_raw:
            tokens = self.tokenizer.encode(itm["text"])[1:-1]
            tokens_arr.append(tokens)
            if max_length < len(tokens):
                max_length = len(tokens)
            itm["etokens"] = self.tokenizer.convert_ids_to_tokens(tokens)
            input_data_raw.append(itm)
        max_length = min(500, max_length)
        for k in range(len(input_data_raw)):
            input_data_raw[k]["etokens"] = input_data_raw[k]["etokens"][:max_length]
        tokens_out = []
        for itm in tokens_arr:
            itm = itm[:max_length]
            itm += [self.TOK_PAD for _ in range(max_length - len(itm))]
            itm = [self.TOK_START] + itm + [self.TOK_END]
            tokens_out.append(itm)
        tokens_var = Variable(torch.LongTensor(tokens_out))
        # padding.
        pad_mask = Variable(torch.FloatTensor(tokens_out))
        pad_mask[pad_mask != float(self.TOK_PAD)] = -1.0
        pad_mask[pad_mask == float(self.TOK_PAD)] = 0.0
        pad_mask = -pad_mask
        # batch_data
        self.batch_data["max_length"] = max_length
        self.batch_data["input_ids"] = tokens_var.to(self.device)
        self.batch_data["pad_mask"] = pad_mask.to(self.device)
        self.batch_data["input_data_raw"] = input_data_raw


class NER_Model(Tagging_Model):
    """NER Model

    Args:
        path_to_model (str, optional): path to the models. Defaults to None.
        device (str, optional): device. Defaults to "cpu".
    """

    def __init__(
        self,
        path_to_model: str = None,
        device: str = "cpu",
    ):
        if path_to_model is None:
            HOME = os.path.expanduser("~")
            MODEL_DIR = "exciton/models/nlp/named_entity_recognition"
            model = "xlm_roberta_large_ner_hrl"
            path_to_model = f"{HOME}/{MODEL_DIR}/{model}"
        label_file = f"{path_to_model}/models/labels.json"
        with open(label_file, "r") as fp:
            labels = json.load(fp)
        n_classes = len(labels)
        self.labels = labels
        self.n_classes = n_classes
        super().__init__(
            path_to_model=path_to_model, n_classes=n_classes, device=device
        )
        self.build_modules()

    def _get_raw_results(
        self, input_data: List[Dict[str, Any]], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get raw results.

        Args:
            input_data (List[Union[str, Dict[str, Any]]]): batch text input.
            top_n (int, optional): top-n. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: output data.
        """
        assert isinstance(input_data, List)
        batch_data = []
        for itm in input_data:
            try:
                itm["text"] = " ".join(itm["text"].split())
            except Exception:
                itm["text"] = ""
            batch_data.append(itm)
        self._build_batch(batch_data)
        with torch.no_grad():
            logits = self._build_pipe()
            logits = torch.softmax(logits, dim=2)
        prob, labels = logits.topk(self.n_classes, dim=2)
        prob = prob.squeeze(2)
        prob = prob.data.cpu().numpy().tolist()
        labels = labels.squeeze(2)
        labels = labels.data.cpu().numpy().tolist()
        torch.cuda.empty_cache()
        output = []
        for k, itm in enumerate(self.batch_data["input_data_raw"]):
            lname = [self.labels[lb[0]] for lb in labels[k]]
            itm["labels"] = lname[1:][: len(itm["etokens"])]
            lcand = []
            for arr in labels[k]:
                out = []
                for lb in arr[:top_n]:
                    out.append(self.labels[lb])
                lcand.append(out)
            lcand = lcand[1:][: len(itm["etokens"])]
            itm["cand_labels"] = lcand
            itm["cand_prob"] = [arr[:top_n] for arr in prob[k]]
            itm["cand_prob"] = itm["cand_prob"][1:][: len(itm["etokens"])]
            output.append(clean_result(itm))

        return output

    def predict(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Annotate texts.

        Args:
            input_data (list): input data.

        Returns:
            List[Dict[str, Any]]: output data.
        """
        batch_size = min(20, len(input_data))
        k = 0
        output = []
        while k * batch_size < len(input_data):
            batch_data = input_data[k * batch_size : (k + 1) * batch_size]
            results = self._get_raw_results(batch_data)
            for itm in results:
                out = {"text": itm["text"], "named_entities": itm["named_entities"]}
                output.append(out)
            k += 1
        return output
