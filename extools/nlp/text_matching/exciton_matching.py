import os
from typing import Any, Dict, List

from transformers import AutoTokenizer


class Text_Matching(object):
    """Text Maching implemented by ExcitonX team.

    Args:
        path_to_model (str, optional): path to model. Defaults to None.
    """

    def __init__(self, path_to_model: str = None) -> None:
        if path_to_model is None:
            HOME = os.path.expanduser("~")
            MODEL_DIR = "exciton/models/nlp/text_matching/exciton_matching"
            path_to_model = f"{HOME}/{MODEL_DIR}"
        self.tokenizer = AutoTokenizer.from_pretrained(f"{path_to_model}/tokenizer")

    @staticmethod
    def _count_tokens(tokens: List[str]) -> Dict[str, int]:
        """Tokens Count

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            Dict[str, int]: Tokens Count.
        """
        counter = {}
        for t in tokens:
            if t in counter.keys():
                counter[t] += 1
            else:
                counter[t] = 1
        return counter

    def predict(self, textA: str, textB: str, lower_case: str = True) -> Dict[str, Any]:
        """Compare Two String.

        Args:
            textA (str): Text A
            textB (str): Text B
            lower_case (str, optional): Convert to lower case. Defaults to True.

        Returns:
            Dict[str, Any]: Scores.
        """
        if lower_case:
            textA = textA.lower().strip()
            textB = textB.lower().strip()
        output = {
            "textA": textA,
            "textB": textB,
            "tokensA": [],
            "tokensB": [],
            "prob_AinB": 0,
            "prob_BinA": 0,
            "score": 0,
        }
        if len(textA) == 0 or len(textB) == 0:
            return output
        # Tokenize
        try:
            tokensA = self.tokenizer.tokenize(textA)
            tokensB = self.tokenizer.tokenize(textB)
        except Exception:
            tokensA = list(textA)
            tokensB = list(textB)
        output["tokensA"] = tokensA
        output["tokensB"] = tokensB
        # Calculate score
        set_a = self._count_tokens(tokensA)
        set_b = self._count_tokens(tokensB)
        match = 0.0
        for token in set_a.keys():
            if token in set_b.keys():
                match += min(set_a[token], set_b[token])
        AinB = match / len(tokensA)
        BinA = match / len(tokensB)
        output["prob_AinB"] = AinB
        output["prob_BinA"] = BinA
        output["score"] = 2 * AinB * BinA / (AinB + BinA)
        return output
