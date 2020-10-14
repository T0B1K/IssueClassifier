import json
import os

import pika

from microservice.celery_app import classify, vectorise

PIKA_AUTO_ACK = os.environ["PIKA_AUTO_ACK"] or True
PIKA_DEFAULT_ROUTING_KEY = os.environ["PIKA_DEFAULT_ROUTING_KEY"] or "Classification.Classify"
PIKA_EXCHANGE_NAME = os.environ["PIKA_EXCHANGE_NAME"] or 'classification_requests'
PIKA_EXCHANGE_TYPE = os.environ["PIKA_EXCHANGE_TYPE"] or 'direct'
PIKA_IS_QUEUE_EXCLUSIVE = os.environ["PIKA_IS_QUEUE_EXCLUSIVE"] or True
PIKA_QUEUE_NAME = os.environ["PIKA_QUEUE_NAME"] or ''
PIKA_RABBITMQ_HOST = os.environ["PIKA_RABBITMQ_HOST"] or 'localhost'


class ICMPikaClient(object):
    """Issue Classifier Microservice Pika RabbitMQ Client

    This is a wrapper class around pika suitable for quickly getting the microservice up and running consuming classification requests. Default values for several configuration options, such as the host for the running RabbitMQ instance, can be found above.
    """

    def __init__(self, routing_keys=PIKA_DEFAULT_ROUTING_KEY):
        self.routing_keys = routing_keys

        self._init_connection()
        self._declare_exchange()
        self._declare_queue()
        self._bind_rkey_to_queue()

    def _init_connection(self):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=PIKA_RABBITMQ_HOST))
        self.channel = connection.channel()

    def _declare_exchange(self):
        self.channel.exchange_declare(
            exchange=PIKA_EXCHANGE_NAME, exchange_type=PIKA_EXCHANGE_TYPE)

    def _declare_queue(self):
        res_queue_declare = self.channel.queue_declare(
            queue=PIKA_QUEUE_NAME, exclusive=PIKA_IS_QUEUE_EXCLUSIVE)
        self.queue = res_queue_declare.method.queue

    def _bind_rkey_to_queue(self):
        self.channel.queue_bind(
            exchange=PIKA_EXCHANGE_NAME, queue=self.queue, routing_key=PIKA_DEFAULT_ROUTING_KEY)

    def handle_issue_request(self, channel, method, properties, message_body):
        issues = json.loads(message_body)
        issues_body_list = [issue["body"] for issue in issues]

        vectorisation_aresult = vectorise.delay(issues_body_list)
        vectorised_issues = vectorisation_aresult.get(
            timeout=5, propagate=False)

        classify.map(vectorised_issues)

    def start_consuming_issue_requests(self):
        self.channel.basic_consume(
            queue=self.queue, on_message_callback=self.
            handle_issue_request, auto_ack=PIKA_AUTO_ACK)
        print("Now consuming issue classification requests. To cancel, press CTRL+C.")
        self.channel.start_consuming()


if __name__ == "__main__":
    pika_client = ICMPikaClient()
    pika_client.start_consuming_issue_requests()
