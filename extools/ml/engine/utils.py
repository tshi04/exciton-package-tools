import random
from typing import Any, Dict, List


def create_batch(
    corpora: List[Dict[str, Any]],
    batch_size: int = 20,
    is_shuffle: bool = False,
) -> List[List[str]]:
    """Split a dataset into batches.

    Args:
        corpora (List[Dict[str, Any]]): Original data.
        batch_size (int, optional): Batch size. Defaults to 20.
        is_shuffle (bool, optional): Shuffle the data in every epoch? Defaults to False.
        is_lower (bool, optional): Convert to lower case? Defaults to False.

    Returns:
        List[List[str]]: batches
    """
    if is_shuffle:
        random.shuffle(corpora)
    # split corpora
    corpora_split = []
    arr = []
    for itm in corpora:
        arr.append(itm)
        if len(arr) == batch_size:
            corpora_split.append(arr)
            arr = []
    if len(arr) > 0:
        corpora_split.append(arr)
        arr = []
    return corpora_split
