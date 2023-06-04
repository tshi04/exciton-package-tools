import json
import logging
import os
from glob import glob
from typing import Any, Dict, List

from exciton.ml.classification.xlm_roberta import Classfication_Model


class Exciton_Relevance(Classfication_Model):
    def __init__(self, path_to_model: str, device: str = "cpu") -> None:
        if path_to_model is None:
            HOME = os.path.expanduser("~")
            MODEL_DIR = "exciton/models/nlp/topic_relevance"
            path_to_model = f"{HOME}/{MODEL_DIR}"
        self.path_to_model = path_to_model
        rel_models = glob(path_to_model)
        for rel_path in rel_models:
            print(rel_path)
        label_file = "{}/labels.json".format(path_to_model)
        with open(label_file, "r") as fp:
            labels = json.load(fp)
        n_classes = len(labels)
        self.labels = labels
        self.n_classes = n_classes
        super().__init__(device=device)

        logging.disable(logging.INFO)
        logging.disable(logging.WARNING)
        self.build_modules()

        self._init_model_parameters()

    def predict(
        self, source: List[Dict[str, Any]], topics: List[Dict[str, Any]] = []
    ) -> List[Dict[str, Any]]:
        """Predict labels.

        Args:
            source (List[Dict[str, Any]]): Source data.
            topics (List[Dict[str, Any]], optional): Topics. Defaults to [].

        Returns:
            List[Dict[str, Any]]: Target labels.
        """
        return 0
