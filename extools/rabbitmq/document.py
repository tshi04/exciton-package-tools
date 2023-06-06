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
