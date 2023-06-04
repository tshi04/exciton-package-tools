import os
from typing import Any, Dict, List, Literal

import torch
from torch.autograd import Variable
from transformers import AutoTokenizer, BertModel, XLMRobertaModel

from exciton.ml.tagging.utils import EncoderRNN


class Tagging_Model(object):
    """Tagging Model.

    Args:
        drop_rate (float, optional): dropout rate. Defaults to 0.1.
        n_classes (int, optional): number of labels. Defaults to 0.
        plm (Literal[&quot;bert&quot;, &quot;xlmroberta&quot;], optional): pretrained language model. Defaults to "xlmroberta".
        device (str, optional): device. Defaults to "cpu".
    """

    def __init__(
        self,
        drop_rate: float = 0.1,
        n_classes: int = 0,
        plm: Literal["bert", "xlmroberta"] = "xlmroberta",
        device: str = "cpu",
    ):
        HOME = os.path.expanduser("~")
        MODEL_DIR = "exciton/models/nlp/pretrained/"

        self.drop_rate = drop_rate
        self.n_classes = n_classes
        self.plm = plm
        self.device = torch.device(device)
        self.train_modules = {}
        self.base_modules = {}
        self.batch_data = {}
        if plm == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{HOME}/{MODEL_DIR}/bert-base-cased/tokenizer"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{HOME}/{MODEL_DIR}/xlm-roberta-base/tokenizer"
            )

    def build_modules(self):
        """Declare all modules in your model."""
        HOME = os.path.expanduser("~")
        MODEL_DIR = "exciton/models/nlp/pretrained/"

        hidden_size = 768
        if self.plm == "bert":
            self.base_modules["embedding"] = BertModel.from_pretrained(
                f"{HOME}/{MODEL_DIR}/bert-base-cased/models",
                output_hidden_states=True,
                output_attentions=True,
            ).to(self.device)
            self.TOK_START = 101
            self.TOK_END = 102
            self.TOK_PAD = 0
        else:
            self.base_modules["embedding"] = XLMRobertaModel.from_pretrained(
                f"{HOME}/{MODEL_DIR}/xlm-roberta-base/models",
                output_hidden_states=True,
                output_attentions=True,
            ).to(self.device)
            self.TOK_START = 0
            self.TOK_END = 2
            self.TOK_PAD = 1
        self.train_modules["encoder"] = EncoderRNN(
            embedding_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=2,
            rnn_network="lstm",
            device=self.device,
        ).to(self.device)
        self.train_modules["classifier"] = torch.nn.Linear(
            hidden_size * 2, self.n_classes
        ).to(self.device)
        self.train_modules["drop"] = torch.nn.Dropout(self.drop_rate).to(self.device)

    def _init_model_parameters(self):
        """load model param"""
        for model_name in self.base_modules:
            self.base_modules[model_name].eval()
            model_file = "{}/{}.model".format(self.path_to_model, model_name)
            self.base_modules[model_name].load_state_dict(
                torch.load(model_file, map_location=lambda storage, loc: storage)
            )
        for model_name in self.train_modules:
            self.train_modules[model_name].eval()
            model_file = "{}/{}.model".format(self.path_to_model, model_name)
            self.train_modules[model_name].load_state_dict(
                torch.load(model_file, map_location=lambda storage, loc: storage)
            )

    def _build_pipe(self):
        """Shared pipe"""
        with torch.no_grad():
            input_emb = self.base_modules["embedding"](
                self.batch_data["input_ids"],
                self.batch_data["pad_mask"],
            )
            input_emb = input_emb[0]
        input_enc, _ = self.train_modules["encoder"](input_emb)
        logits = self.train_modules["classifier"](
            self.train_modules["drop"](torch.relu(input_enc))
        )
        return logits

    def _tokenize_text(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """tokenize text.

        Args:
            input_data (Dict[str, Any]): input data.

        Returns:
            Dict[str, Any]: output.
        """
        if "tokens" in input_data and "labels" in input_data:
            tokens = input_data["tokens"]
            labels = input_data["labels"]
            assert len(tokens) == len(labels)
            out_tokens = []
            out_labels = []
            for k, tok in enumerate(tokens):
                rbtok = self.tokenizer.encode(tok)[1:-1]
                rblab = []
                lab = labels[k]
                for j in range(len(rbtok)):
                    if j == 0:
                        rblab.append(lab)
                    else:
                        if lab[0] == "B":
                            rblab.append(f"I{lab[1:]}")
                        else:
                            rblab.append(lab)
                out_tokens.extend(rbtok)
                out_labels.extend(rblab)
            return {"tokens": out_tokens, "labels": out_labels}
        else:
            out = self.tokenizer.encode(input_data["text"])[1:-1]
            return {"tokens": out}

    def _build_batch(self, batch_raw: List[Any]):
        """Build Batch

        Args:
            batch_data (List[Any]): batch data.
        """
        tokens_arr = []
        input_data_raw = []
        max_length = 0
        for itm in batch_raw:
            out = self._tokenize_text(input_data=itm)
            tokens_arr.append(out["tokens"])
            if max_length < len(out["tokens"]):
                max_length = len(out["tokens"])
            itm["etokens"] = self.tokenizer.convert_ids_to_tokens(out["tokens"])
            if "labels" in out:
                itm["elabels"] = out["labels"]
            input_data_raw.append(itm)
        max_length = min(500, max_length)
        for k in range(len(input_data_raw)):
            input_data_raw[k]["etokens"] = input_data_raw[k]["etokens"][:max_length]
            if "elabels" in input_data_raw[k]:
                input_data_raw[k]["elabels"] = input_data_raw[k]["elabels"][:max_length]
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
