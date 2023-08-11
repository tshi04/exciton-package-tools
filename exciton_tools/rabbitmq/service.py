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
    """send messages to an exchange

    Args:
        host (str): host
        port (str): port
        virtual_host (str): virtual host
        username (str): username
        password (str): password
        exchange_name (str): exchange name
        exchange_type (str): exchange type
        routing_key (str): rounting key
        messages (List[str]): message
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
