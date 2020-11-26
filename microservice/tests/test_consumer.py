import logging
from time import sleep
from typing import Dict, List

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties
import ujson

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

ROUTING_KEY = "Classification.Results"
EXCHANGE_NAME = "classification"
EXCHANGE_TYPE = "direct"
QUEUE_NAME = "issue_classifier_output"
RABBITMQ_HOST = "localhost"

connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
channel = connection.channel()
channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE)
queue_declaration = channel.queue_declare(queue=QUEUE_NAME, durable=True)
queue = queue_declaration.method.queue

first_issue: bool = True
response_count: int = 0


def handle_respone(
    channel: BlockingChannel,
    method_frame: Basic.Deliver,
    header_frame: BasicProperties,
    message_body: bytes,
) -> None:
    global first_issue
    global response_count

    if first_issue:
        logging.info("First issue has arrived!")
        first_issue = False

    response_count = response_count + 1
    logging.info("Response number " + str(response_count) + " has arrived.")


channel.basic_consume(
    queue=queue,
    on_message_callback=handle_respone,
    auto_ack=True,
)


if __name__ == "__main__":
    logging.info("Now consuming responses...")
    channel.start_consuming()