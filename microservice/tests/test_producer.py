import logging
from time import sleep
from typing import Dict, List

import pika
import ujson

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

ROUTING_KEY = "Classification.Classify"
EXCHANGE_NAME = "classification"
EXCHANGE_TYPE = "direct"
RABBITMQ_HOST = "localhost"

connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
channel = connection.channel()
channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE)

with open("issues/bug.json") as file:
    data = ujson.loads(file.read())

text_list: list = list(map(lambda entry: entry["text"], data))[:4000]
indexed_issues: List[Dict] = [
    {"index": index, "body": body} for index, body in enumerate(text_list)
]

i: int = 0
delta: int = 2000
delay: int = 60
while i + delta < len(indexed_issues):
    print("Round " + str(i) + " of sending")
    start: int = i
    stop: int = i + delta
    current_issues = indexed_issues[start:stop]
    current_issues_in_json = ujson.dumps(current_issues).encode("utf-8")
    if i == 0:
        logging.info("Sending first batch of issues. Count: " + str(delta))
    channel.basic_publish(
        exchange=EXCHANGE_NAME, routing_key=ROUTING_KEY, body=current_issues_in_json
    )
    i += delta
    sleep(delay)

print("Issue classification test complete.")