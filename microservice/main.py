import json

from celery import chunks
import pika

from celery_app import classify, vectorise

AUTO_ACK = True
DEFAULT_ROUTING_KEYS = ["Classification.Classify"]
EXCHANGE_NAME = 'classification_requests'
EXCHANGE_TYPE = 'direct'
IS_QUEUE_EXCLUSIVE = True
QUEUE_NAME = ''
RABBITMQ_HOST = 'localhost'


class ICMPikaClient(object):
    """Issue Classifier Microservice Pika RabbitMQ Client

    This is a wrapper class around pika suitable for quickly getting the microservice up and running consuming classification requests. Default values for several configuration options, such as the host for the running RabbitMQ instance, can be found above.
    """

    def __init__(self, routing_keys=DEFAULT_ROUTING_KEYS):
        self.routing_keys = routing_keys

        self._init_connection()
        self._declare_exchange()
        self._declare_queue()
        self._bind_rkeys_to_queue()

    def _init_connection(self):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST))
        self.channel = connection.channel()

    def _declare_exchange(self):
        self.channel.exchange_declare(
            exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE)

    def _declare_queue(self):
        res_queue_declare = self.channel.queue_declare(
            queue=QUEUE_NAME, exclusive=IS_QUEUE_EXCLUSIVE)
        self.queue = res_queue_declare.method.queue

    def _bind_rkeys_to_queue(self):
        for routing_key in self.routing_keys:
            self.channel.queue_bind(
                exchange=EXCHANGE_NAME, queue=self.queue, routing_key=routing_key)

    def handle_issue_request(self, channel, method, properties, message_body):
        issues = json.loads(message_body)
        issues_body_list = [issue["body"] for issue in issues]

        vectorisation_aresult = vectorise.delay(issues_body_list)
        vectorised_issues = vectorisation_aresult.get(timeout=5, propagate=False)
        
        classify.map(vectorised_issues)

    def start_consuming_issue_requests(self):
        self.channel.basic_consume(
            queue=self.queue, on_message_callback=self.
            handle_issue_request, auto_ack=AUTO_ACK)
        print("Now consuming issue classification requests. To cancel, press CTRL+C.")
        self.channel.start_consuming()


if __name__ == "__main__":
    pika_client = ICMPikaClient()
    pika_client.start_consuming_issue_requests()
