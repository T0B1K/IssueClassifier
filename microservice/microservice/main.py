import json
import os

import pika

from microservice.celery_app import classify, vectorise

PIKA_AUTO_ACK = os.getenv("PIKA_AUTO_ACK", True)
PIKA_INPUT_ROUTING_KEY = os.getenv(
    "PIKA_INPUT_ROUTING_KEY", "Classification.Classify")
PIKA_OUTPUT_ROUTING_KEY = os.getenv(
    "PIKA_OUTPUT_ROUTING_KEY", "Classification.Results")
PIKA_EXCHANGE_NAME = os.getenv("PIKA_EXCHANGE_NAME", 'classification')
PIKA_EXCHANGE_TYPE = os.getenv("PIKA_EXCHANGE_TYPE", 'direct')
PIKA_INPUT_QUEUE_NAME = os.getenv(
    "PIKA_INPUT_QUEUE_NAME", 'ic_microservice_input_queue')
PIKA_OUTPUT_QUEUE_NAME = os.getenv(
    "PIKA_OUTPUT_QUEUE_NAME", 'ic_microservice_output_queue')
PIKA_RABBITMQ_HOST = os.getenv("PIKA_RABBITMQ_HOST", 'localhost')
VECTORISER_QUEUE = os.getenv('VECTORISER_QUEUE', "vectorise_queue")
CLASSIFIER_QUEUE = os.getenv('CLASSIFIER_QUEUE', "classify_queue")


class ICMPikaClient(object):
    """Issue Classifier Microservice Pika RabbitMQ Client

    This is a wrapper class around pika suitable for quickly getting the microservice up and running consuming classification requests. Default values for several configuration options, such as the host for the running RabbitMQ instance, can be found above.
    """

    def __init__(self, routing_key=PIKA_INPUT_ROUTING_KEY):
        self.routing_key = routing_key

        self._init_connection()
        self._declare_exchange()
        self._declare_input_queue()
        self._declare_output_queue()
        self._bind_rkeys_to_queues()

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

    def _declare_input_queue(self):
        """Declare a (new if not previously existing) queue with the connected RabbitMQ instance.

        Uses the follwing environment variables:
            - PIKA_QUEUE_NAME: The name of the queue to be declared.
            - PIKA_IS_QUEUE_EXCLUSIVE: Whether the queue should be declared as exclusive, 
        i.e. whether the queue can only be used by the channel of the declaring running pika client.
        """
        input_queue = self.channel.queue_declare(
            queue=PIKA_INPUT_QUEUE_NAME)
        self.input_queue = input_queue.method.queue

    def _declare_output_queue(self):
        """Declare a (new if not previously existing) queue with the connected  RabbitMQ instance.

        Uses the follwing environment variables:
            - PIKA_QUEUE_NAME: The name of the queue to be declared.
            - PIKA_IS_QUEUE_EXCLUSIVE: Whether the queue should be declared as  exclusive, 
        i.e. whether the queue can only be used by the channel of the declaring     running pika client.
        """
        output_queue = self.channel.queue_declare(
            queue=PIKA_OUTPUT_QUEUE_NAME)
        self.output_queue = output_queue.method.queue

    def _bind_rkeys_to_queues(self):
        """Bind queue to the exchange declared in the pika client using a routing key.

        Uses the following environment variables:
            - PIKA_EXCHANGE_NAME: The name of the exchange to be 
            - PIKA_DEFAULT_ROUTING_KEY: The routing key to bind the queue and exchange on.
        """
        self.channel.queue_bind(
            exchange=PIKA_EXCHANGE_NAME,
            queue=self.input_queue,
            routing_key=PIKA_INPUT_ROUTING_KEY)
        self.channel.queue_bind(
            exchange=PIKA_EXCHANGE_NAME,
            queue=self.output_queue,
            routing_key=PIKA_OUTPUT_ROUTING_KEY
        )

    def _return_classification_results(self, results):
        self.channel.basic_publish(
            exchange=PIKA_EXCHANGE_NAME,
            routing_key=PIKA_OUTPUT_ROUTING_KEY,
            body=json.dumps(list(results))
        )

    def _handle_issue_request(self, channel, method, properties, message_body):
        """Callback function for incoming issue classification request messages.

        Args:
            channel ([type]): The channel over which the request came in.
            method ([type]): The method of the incoming message.
            properties ([type]): Various properties of the incoming message.
            message_body ([type]): The payload of the incoming message.
        """
        issues = json.loads(message_body)
        issues_body_list = [issue["body"] for issue in issues]
        print(issues_body_list)

        vectorised_issues = vectorise.apply_async(
            (issues_body_list,), queue=VECTORISER_QUEUE).get(
            timeout=10, propagate=False)
        print(vectorised_issues)

        classified_issues = classify.apply_async((vectorised_issues,),
                                                 queue=CLASSIFIER_QUEUE).get(timeout=120, propagate=False)
        print(classified_issues)
        
        self._return_classification_results(classified_issues)

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
            queue=self.input_queue, on_message_callback=self.
            _handle_issue_request, auto_ack=PIKA_AUTO_ACK)
        print("Now consuming issue classification requests. To cancel, press CTRL+C.")
        self.channel.start_consuming()


if __name__ == "__main__":
    pika_client = ICMPikaClient()
    pika_client.start_consuming_issue_requests()
