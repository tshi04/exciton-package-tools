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
    """Send messages to an exchange.

    Args:
        host (str): host
        port (str): port (amqp)
        virtual_host (str): virtual host
        username (str): username
        password (str): password
        exchange_name (str): exchange name
        exchange_type (str): exchange type
        routing_key (str): rounting key
        messages (List[str]): list of messages.
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


def get_rmq_queue_length(
    host: str,
    port: str,
    virtual_host: str,
    username: str,
    password: str,
    queue_name: str,
) -> int:
    """Get the length of RabbitMQ queue.

    Args:
        host (str): host
        port (str): port (amqp)
        virtual_host (str): virtual host
        username (str): username
        password (str): password
        queue_name (str): queue name
    """
    # RABBITMQ
    credentials = pika.PlainCredentials(
        username=username,
        password=password,
    )
    connectparam = pika.ConnectionParameters(
        host=host,
        port=port,
        virtual_host=virtual_host,
        credentials=credentials,
        heartbeat=0,
    )
    connection = pika.BlockingConnection(connectparam)
    channel = connection.channel()
    resp = channel.queue_declare(queue=queue_name, durable=True)
    channel.close()

    return resp.method.message_count
