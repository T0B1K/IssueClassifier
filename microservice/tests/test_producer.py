import logging
from time import sleep
from typing import Dict, List

import pika
import ujson

ROUTING_KEY = "Classification.Classify"
EXCHANGE_NAME = "classification"
EXCHANGE_TYPE = "direct"
RABBITMQ_HOST = "localhost"

connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
channel = connection.channel()
channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE)

with open("issues/enhancement.json") as file:
    data = ujson.loads(file.read())

text_list: list = list(map(lambda entry: entry["text"], data))[:4000]
indexed_issues: List[Dict] = [
    {"index": index, "body": body} for index, body in enumerate(text_list)
]

i: int = 0
while i + 200 < len(indexed_issues):
    print("Round " + str(i) + " of sending")
    start: int = i
    stop: int = i + 200
    current_issues = indexed_issues[start:stop]
    current_issues_in_json = ujson.dumps(current_issues).encode("utf-8")
    channel.basic_publish(
        exchange=EXCHANGE_NAME, routing_key=ROUTING_KEY, body=current_issues_in_json
    )
    i += 200
    sleep(30)

print("Issue classification test complete.")