import json
from typing import Any, List, Literal

import torch
from torch.autograd import Variable

from exciton.ml.engine.end2end import End2EndBase

from .model import Tagging_Model


class Tagging_Trainer(Tagging_Model, End2EndBase):
    """Tagging trainer

    Args:
        n_classes (str): number of labels.
        working_dir (str): working directory.
        plm (Literal[&quot;bert&quot;, &quot;xlmroberta&quot;], optional): pretrained language model. Defaults to "xlmroberta".
        drop_rate (float, optional): dropout rate. Defaults to 0.1.
        device (str, optional): device. Defaults to "cpu".
    """

    def __init__(
        self,
        n_classes: str,
        working_dir: str,
        plm: Literal["bert", "xlmroberta"] = "xlmroberta",
        drop_rate: float = 0.1,
        device: str = "cpu",
    ):
        Tagging_Model.__init__(
            self, n_classes=n_classes, plm=plm, drop_rate=drop_rate, device=device
        )
        End2EndBase.__init__(self, working_dir=working_dir)
        self.n_classes = n_classes
        self.loss_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(
            torch.device(device)
        )

    def _build_batch(self, batch_raw: List[Any]):
        """Build Batch

        Args:
            batch_data (List[Any]): batch data.
        """
        Tagging_Model._build_batch(self, batch_raw=batch_raw)
        labels_arr = []
        batch_tmp = self.batch_data["input_data_raw"]
        label_mapping = ""
        for itm in batch_tmp:
            tmp_labels = []
            for lb in itm["elabels"]:
                tmp_labels.append(itm["label_mapping"].index(lb))
            labels_arr.append(tmp_labels)
            label_mapping = itm["label_mapping"]

        max_length = self.batch_data["max_length"]
        labels_out = []
        for itm in labels_arr:
            itm = itm[:max_length]
            itm += [-1 for _ in range(max_length - len(itm))]
            itm = [-1] + itm + [-1]
            labels_out.append(itm)
        label_var = Variable(torch.LongTensor(labels_out))
        self.batch_data["labels"] = label_var.to(self.device)
        with open(f"{self.working_dir}/labels.json", "w") as fout:
            json.dump(label_mapping, fout)

    def _build_pipeline(self):
        """Data pipeline"""
        logits = self._build_pipe()
        logits = logits.contiguous().view(-1, self.n_classes)
        loss = self.loss_criterion(logits, self.batch_data["labels"].view(-1))
        return loss

    def build_modules(self):
        Tagging_Model.build_modules(self)
        self._init_base_modules_params()
