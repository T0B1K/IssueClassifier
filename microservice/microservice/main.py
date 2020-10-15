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
VECTORISER_QUEUE = os.environ['VECTORISER_QUEUE'] or "vectorise_queue"
CLASSIFIER_QUEUE = os.environ['CLASSIFIER_QUEUE'] or "classify_queue"


class ICMPikaClient(object):
    """Issue Classifier Microservice Pika RabbitMQ Client

    This is a wrapper class around pika suitable for quickly getting the microservice up and running consuming classification requests. Default values for several configuration options, such as the host for the running RabbitMQ instance, can be found above.
    """

    def __init__(self, routing_key=PIKA_DEFAULT_ROUTING_KEY):
        self.routing_key = routing_key

        self._init_connection()
        self._declare_exchange()
        self._declare_queue()
        self._bind_rkey_to_queue()

    def _init_connection(self):
        """Initialise connection with the RabbitMQ instance

        Uses the follwing environment variable:
            - PIKA_RABBITMQ_HOST: Hostname of the running RabbitMQ instance to connect to.
        """
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=PIKA_RABBITMQ_HOST))
        self.channel = connection.channel()

    def _declare_exchange(self):
        """Declare a (new if not previously existing) exchange with the connected RabbitMQ instance.

        Uses the follwing environment variables:
            - PIKA_EXCHANGE_NAME: The name of the exchange to be declared.
            - PIKA_EXCHANGE_TYPE: The type of the exchange to be declared.
        """
        self.channel.exchange_declare(
            exchange=PIKA_EXCHANGE_NAME, exchange_type=PIKA_EXCHANGE_TYPE)

    def _declare_queue(self):
        """Declare a (new if not previously existing) queue with the connected RabbitMQ instance.

        Uses the follwing environment variables:
            - PIKA_QUEUE_NAME: The name of the queue to be declared.
            - PIKA_IS_QUEUE_EXCLUSIVE: Whether the queue should be declared as exclusive, 
        i.e. whether the queue can only be used by the channel of the declaring running pika client.
        """
        res_queue_declare = self.channel.queue_declare(
            queue=PIKA_QUEUE_NAME, exclusive=PIKA_IS_QUEUE_EXCLUSIVE)
        self.queue = res_queue_declare.method.queue

    def _bind_rkey_to_queue(self):
        """Bind queue to the exchange declared in the pika client using a routing key.

        Uses the following environment variables:
            - PIKA_EXCHANGE_NAME: The name of the exchange to be 
            - PIKA_DEFAULT_ROUTING_KEY: The routing key to bind the queue and exchange on.
        """
        self.channel.queue_bind(
            exchange=PIKA_EXCHANGE_NAME, queue=self.queue, routing_key=PIKA_DEFAULT_ROUTING_KEY)

    def handle_issue_request(self, channel, method, properties, message_body):
        # FIXME Give types of method paramteres
        """Callback function for incoming issue classification request messages.

        Args:
            channel ([type]): The channel over which the request came in.
            method ([type]): The method of the incoming message.
            properties ([type]): Various properties of the incoming message.
            message_body ([type]): The payload of the incoming message.
        """
        issues = json.loads(message_body)
        issues_body_list = [issue["body"] for issue in issues]

        vectorisation_aresult = vectorise.apply_async((issues_body_list,), queue=VECTORISER_QUEUE)
        vectorised_issues = vectorisation_aresult.get(
            timeout=5, propagate=False)

        classify.map(vectorised_issues).apply_async(queue=CLASSIFIER_QUEUE)

    def start_consuming_issue_requests(self):
        """The (blocking) request consumption method.

        Once the initialisation phase is completed successfully, the pika client simply waits
        for incoming issue classification requests in the form of AMQP messages to hand over 
        to the callback function handle_issue_request, which in turn passes the message using
        celery to its workers for processing.

        Uses the following environment variable:
            - PIKA_AUTO_ACK: Whether the client should acknowledge all incoming requests 
            to inform the RabbitMQ instance of the successful reception of the message.
        """
        self.channel.basic_consume(
            queue=self.queue, on_message_callback=self.
            handle_issue_request, auto_ack=PIKA_AUTO_ACK)
        print("Now consuming issue classification requests. To cancel, press CTRL+C.")
        self.channel.start_consuming()


if __name__ == "__main__":
    pika_client = ICMPikaClient()
    pika_client.start_consuming_issue_requests()
