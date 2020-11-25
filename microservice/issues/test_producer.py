import json
import logging
import pika

INPUT_ROUTING_KEY = ["Classification.Classify"]
EXCHANGE_NAME = "classification"
EXCHANGE_TYPE = "direct"
RABBITMQ_HOST = "localhost"

connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
channel = connection.channel()

channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE)

routing_key = INPUT_ROUTING_KEY[0]

with open("enhancement.json") as file:
    data = json.loads(file.read())

documents: list = list(map(lambda entry: entry["text"], data))[:4000]
logging.warning("--------> documents.size: {} ---------".format(len(documents)))

array: list = documents
i: int = 0
while i * 100 < len(array):
    idx = i * 100
    sendingArray = array[idx : idx + 100]
    logging.warn("{}: sending: {}".format(idx, len(sendingArray)))
    serialized = json.dumps(sendingArray)
    channel.basic_publish(
        exchange="classification",
        routing_key="Classification.Classify",
        body=json.dumps(list(serialized)).encode("utf-8"),
    )
    i = i + 1
