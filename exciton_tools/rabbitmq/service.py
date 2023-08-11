from typing import List

import pika
import urllib3

urllib3.disable_warnings()


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
    """_summary_

    Args:
        host (str): _description_
        port (str): _description_
        virtual_host (str): _description_
        username (str): _description_
        password (str): _description_
        exchange_name (str): _description_
        exchange_type (str): _description_
        routing_key (str): _description_
        messages (List[str]): _description_
    """
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
