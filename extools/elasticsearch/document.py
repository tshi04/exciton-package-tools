from typing import Any, Dict

from elasticsearch import Elasticsearch
import urllib3
from typing import Any, Dict, List

import pika

urllib3.disable_warnings()

def get_all_docs(
    client: Elasticsearch,
    index: str,
    query: Dict[str, Any] = {"match_all": {}},
    get_source: bool = True,
) -> Dict[str, Any]:
    
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


def send_messages_to_exchange(
    host: str,
    port: str,
    virtual_host: str,
    username: str,
    password: str,
    exchange_name: str,
    exchange_type: str,
    routing_key: str,
    messages: List[str],
):
    # rabbitmq
    credentials = pika.PlainCredentials(username=username, password=password)
    connectparam = pika.ConnectionParameters(
        host=host,
        port=port,
        virtual_host=virtual_host,
        credentials=credentials,
    )
    connection = pika.BlockingConnection(connectparam)
    channel = connection.channel()
    channel.exchange_declare(
        exchange=exchange_name,
        exchange_type=exchange_type,
    )
    for message in messages:
        channel.basic_publish(
            exchange=exchange_name,
            routing_key=routing_key,
            body=message,
        )
    connection.close()