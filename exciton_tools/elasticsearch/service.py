import gzip
import json
from typing import Any, Dict

import urllib3
from elasticsearch import Elasticsearch

urllib3.disable_warnings()


def dump_data_json(
    client: Elasticsearch,
    index_name: str,
    file_name: str,
    path_to_file: str = "/tmp",
):
    """Dump data to json gzip.

    Args:
        client (Elasticsearch): client.
        index_name (str): index name.
        file_name (str): local file name.
        path_to_file (str, optional): path to the file. Defaults to "/tmp".
    """
    docs = get_all_docs(client=client, index=index_name)
    file_name = f"{path_to_file}/{file_name}"
    fout = gzip.open(file_name, "wt")
    for itm in docs:
        json.dump(itm, fout)
        fout.write("\n")
    fout.close()


def get_all_docs(
    client: Elasticsearch,
    index: str,
    query: Dict[str, Any] = {"match_all": {}},
    get_source: bool = True,
) -> Dict[str, Any]:
    """Get all documents of an index that match the query.

    Args:
        client (Elasticsearch): client
        index (str): index name.
        query (_type_, optional): query. Defaults to {"match_all": {}}.
        get_source (bool, optional): if get source or not. Defaults to True.

    Returns:
        Dict[str, Any]: _description_

    Yields:
        Iterator[Dict[str, Any]]: _description_
    """
    old_scroll_id = None
    resp = client.search(
        index=index, query=query, size=200, scroll="30s", ignore=[400, 404]
    )
    old_scroll_id = resp["_scroll_id"]
    for itm in resp["hits"]["hits"]:
        out = {key: itm[key] for key in itm if key != "_source"}
        if get_source:
            out["_source"] = itm["_source"]
        yield out
    while True:
        resp = client.scroll(scroll_id=old_scroll_id, scroll="30s")
        if len(resp["hits"]["hits"]) == 0:
            break
        for itm in resp["hits"]["hits"]:
            out = {key: itm[key] for key in itm if key != "_source"}
            if get_source:
                out["_source"] = itm["_source"]
            yield out

        old_scroll_id = resp["_scroll_id"]
        if old_scroll_id != resp["_scroll_id"]:
            print("NEW SCROLL ID:", resp["_scroll_id"])
