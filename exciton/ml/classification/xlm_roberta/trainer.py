import json
from typing import Any, List

import torch
from torch.autograd import Variable

from exciton.ml.engine.end2end import End2EndBase

from .model import Classfication_Model


class Classification_Trainer(Classfication_Model, End2EndBase):
    """Tagging trainer

    Args:
        n_classes (str): number of labels.
        working_dir (str): working directory.
        drop_rate (float, optional): dropout rate. Defaults to 0.1.
        device (str, optional): device. Defaults to "cpu".
    """

    def __init__(
        self,
        n_classes: str,
        working_dir: str,
        drop_rate: float = 0.1,
        device: str = "cpu",
    ):
        Classfication_Model.__init__(
            self, n_classes=n_classes, drop_rate=drop_rate, device=device
        )
        End2EndBase.__init__(self, working_dir=working_dir)
        self.n_classes = n_classes
        self.loss_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(
            torch.device(device)
        )

    def _build_batch(self, batch_raw: List[Any]):
        """Build Batch

        Args:
            batch_raw (List[Any]): batch data.
        """
        Classfication_Model._build_batch(self, batch_raw=batch_raw)

        label_arr = []
        for itm in batch_raw:
            label_arr.append(itm["label"])
        label_var = Variable(torch.LongTensor(label_arr))
        self.batch_data["label"] = label_var.to(self.device)
        label_mapping = batch_raw[0]["label_mapping"]
        with open(f"{self.working_dir}/labels.json", "w") as fout:
            json.dump(label_mapping, fout)

    def _build_pipeline(self):
        """Data pipeline"""
        input_emb = self._build_pipe()
        logits = self._build_pipe_classifier(input_emb)
        logits = logits.contiguous().view(-1, self.n_classes)
        loss = self.loss_criterion(logits, self.batch_data["label"].view(-1))
        return loss

    def build_modules(self):
        Classfication_Model.build_modules(self)
        Classfication_Model.build_classifier(self)
        self._init_base_modules_params()
