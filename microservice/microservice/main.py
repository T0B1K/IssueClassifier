import json
import os
from typing import Any, Optional

from pika import ConnectionParameters
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection

from microservice.ms_celery.celery import app as celery_app
from microservice.ms_celery.tasks import ClassifyNodeTask

PIKA_AUTO_ACK: bool = bool(os.getenv("PIKA_AUTO_ACK", True))
PIKA_INPUT_ROUTING_KEY: str = os.getenv(
    "PIKA_INPUT_ROUTING_KEY", "Classification.Classify"
)
PIKA_OUTPUT_ROUTING_KEY: str = os.getenv(
    "PIKA_OUTPUT_ROUTING_KEY", "Classification.Results"
)
PIKA_EXCHANGE_NAME: str = os.getenv("PIKA_EXCHANGE_NAME", "classification")
PIKA_EXCHANGE_TYPE: str = os.getenv("PIKA_EXCHANGE_TYPE", "direct")
PIKA_INPUT_QUEUE_NAME: str = os.getenv(
    "PIKA_INPUT_QUEUE_NAME", "ic_microservice_input_queue"
)
PIKA_OUTPUT_QUEUE_NAME: str = os.getenv(
    "PIKA_OUTPUT_QUEUE_NAME", "ic_microservice_output_queue"
)
PIKA_RABBITMQ_HOST: str = os.getenv("PIKA_RABBITMQ_HOST", "localhost")
CLASSIFY_QUEUE: str = os.getenv("CLASSIFY_QUEUE", "classify_queue")


class ICMPikaClient(object):
    """Issue Classifier Microservice Pika RabbitMQ Client.

    This is a wrapper class around pika suitable for quickly
    getting the microservice up and running consuming classification requests.

    Default values for several configuration options, such as
    the host for the running RabbitMQ instance, can be found above.
    """

    def __init__(self, routing_key: str = PIKA_INPUT_ROUTING_KEY) -> None:
        """Initialise the Issue Classification microservice pika client.

        Args:
            routing_key (str, optional): [description]. Defaults to PIKA_INPUT_ROUTING_KEY.
        """
        self.routing_key: str = routing_key

        self._init_connection()
        self._declare_exchange()
        self._declare_input_queue()
        self._declare_output_queue()
        self._bind_rkeys_to_queues()

    def _init_connection(self):
        """Initialise connection with the RabbitMQ instance.

        Uses the follwing environment variable:
            - PIKA_RABBITMQ_HOST: Hostname of the running RabbitMQ instance to connect to.
        """
        connection = BlockingConnection(ConnectionParameters(host=PIKA_RABBITMQ_HOST))
        self.channel: BlockingChannel = connection.channel()

    def _declare_exchange(self):
        """Declare a (new if not previously existing) exchange with the connected RabbitMQ instance.

        Uses the follwing environment variables:
            - PIKA_EXCHANGE_NAME: The name of the exchange to be declared.
            - PIKA_EXCHANGE_TYPE: The type of the exchange to be declared.
        """
        self.channel.exchange_declare(
            exchange=PIKA_EXCHANGE_NAME, exchange_type=PIKA_EXCHANGE_TYPE
        )

    def _declare_input_queue(self):
        """Declare a (new if not previously existing) queue with the connected RabbitMQ instance.

        Uses the follwing environment variables:
            - PIKA_QUEUE_NAME: The name of the queue to be declared.
            - PIKA_IS_QUEUE_EXCLUSIVE: Whether the queue should be declared as exclusive,
        i.e. whether the queue can only be used by the channel of the declaring running pika client.
        """
        input_queue: Any = self.channel.queue_declare(queue=PIKA_INPUT_QUEUE_NAME)
        self.input_queue: Optional[str] = input_queue.method.queue

    def _declare_output_queue(self):
        """Declare a (new if not previously existing) queue with the connected  RabbitMQ instance.

        Uses the follwing environment variables:
            - PIKA_QUEUE_NAME: The name of the queue to be declared.
            - PIKA_IS_QUEUE_EXCLUSIVE: Whether the queue should be declared as  exclusive,
        i.e. whether the queue can only be used by the channel of the declaring     running pika client.
        """
        output_queue = self.channel.queue_declare(queue=PIKA_OUTPUT_QUEUE_NAME)
        self.output_queue: Optional[str] = output_queue.method.queue

    def _bind_rkeys_to_queues(self):
        """Bind queue to the exchange declared in the pika client using a routing key.

        Uses the following environment variables:
            - PIKA_EXCHANGE_NAME: The name of the exchange to be
            - PIKA_DEFAULT_ROUTING_KEY: The routing key to bind the queue and exchange on.
        """
        if self.input_queue is not None:
            self.channel.queue_bind(
                exchange=PIKA_EXCHANGE_NAME,
                queue=self.input_queue,
                routing_key=PIKA_INPUT_ROUTING_KEY,
            )
        if self.output_queue is not None:
            self.channel.queue_bind(
                exchange=PIKA_EXCHANGE_NAME,
                queue=self.output_queue,
                routing_key=PIKA_OUTPUT_ROUTING_KEY,
            )

    def _return_classification_results(self, results):
        """Return classification results back to the output queue.

        Args:
            results ([type]): [description]
        """
        self.channel.basic_publish(
            exchange=PIKA_EXCHANGE_NAME,
            routing_key=PIKA_OUTPUT_ROUTING_KEY,
            body=json.dumps(list(results)).encode("utf-8"),
        )

    def _handle_issue_request(self, channel, method, properties, message_body):
        """Handle an incoming issue request.

        Args:
            channel ([type]): The channel over which the request came in.
            method ([type]): The method of the incoming message.
            properties ([type]): Various properties of the incoming message.
            message_body ([type]): The payload of the incoming message.
        """
        issues = json.loads(message_body)
        issues_body_list = [issue["body"] for issue in issues]

        classify_node_task = celery_app.register_task(ClassifyNodeTask())  # type: ignore
        classified_issues = classify_node_task.apply_async(
            (issues_body_list,), queue=CLASSIFY_QUEUE
        ).get(timeout=120, propagate=False)

        self._return_classification_results(classified_issues)

    def start_consuming_issue_requests(self):
        """Begins consuming issue requests for processing.

        Once the initialisation phase is completed successfully, the pika client simply waits
        for incoming issue classification requests in the form of AMQP messages to hand over
        to the callback function handle_issue_request, which in turn passes the message using
        celery to its workers for processing.

        Uses the following environment variable:
            - PIKA_AUTO_ACK: Whether the client should acknowledge all incoming requests
            to inform the RabbitMQ instance of the successful reception of the message.
        """
        self.channel.basic_consume(
            queue=self.input_queue,
            on_message_callback=self._handle_issue_request,
            auto_ack=PIKA_AUTO_ACK,
        )
        print("Now consuming issue classification requests. To cancel, press CTRL+C.")
        self.channel.start_consuming()


if __name__ == "__main__":
    pika_client = ICMPikaClient()
    pika_client.start_consuming_issue_requests()
