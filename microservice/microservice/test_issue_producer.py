import pika
import sys

DEFAULT_ROUTING_KEYS = ["Classification.Classify"]
EXCHANGE_NAME = "classification"
EXCHANGE_TYPE = "direct"
RABBITMQ_HOST = "localhost"

connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
channel = connection.channel()

channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE)

routing_keys = sys.argv[1] if len(sys.argv) > 1 else "Classification.Classify"
message = " ".join(sys.argv[2:]) or "Hello World!"
channel.basic_publish(exchange=EXCHANGE_NAME, routing_key=routing_keys, body=message)
print(" [x] Sent %r:%r" % (routing_keys, message))

connection.close()
