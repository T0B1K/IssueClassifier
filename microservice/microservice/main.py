"""The pika client of the Issue Classifier Microservice.

This represents the primary entrypoint into the microservice. Issues to be
classified are to be pushed to RabbitMQ. Specifically, to the appropriate
exchange using the appropriate routing key. Once an issue is received, it's then
forwarded to Celery (more precisely: to workers under Celery's control) for
processing.

Once classification is finished, they are returned by the leaf nodes of the
classification tree back to RabbitMQ. Note that, in order to return results as
soon as possible, classifications are not aggregated. Instead, they are returned
back with their corresponding indices, which would serve in mapping the
classifications to their respective issues.

This necessities the use of unique keys for each classification request. Failure
to do so does not result in incorrect results, but could make it essentially
impossible to correctly map the classification results back to the issue bodies.
"""
import logging
from os import getenv
from typing import Any, List, Optional

from pika import ConnectionParameters
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from pika.spec import Basic, BasicProperties
from pydantic.tools import parse_raw_as

from microservice.classifier_celery.tasks import vectorise_issues
from microservice.models.models import IndexedIssue

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

# Environment variables used throughout this module
PIKA_AUTO_ACK: bool = bool(getenv("PIKA_AUTO_ACK", True))
PIKA_INPUT_ROUTING_KEY: str = getenv(
    "PIKA_INPUT_ROUTING_KEY", "Classification.Classify"
)
PIKA_OUTPUT_ROUTING_KEY: str = getenv(
    "PIKA_OUTPUT_ROUTING_KEY", "Classification.Results"
)
PIKA_EXCHANGE_NAME: str = getenv("PIKA_EXCHANGE_NAME", "classification")
PIKA_EXCHANGE_TYPE: str = getenv("PIKA_EXCHANGE_TYPE", "direct")
PIKA_INPUT_QUEUE_NAME: str = getenv(
    "PIKA_INPUT_QUEUE_NAME", "ic_microservice_input_queue"
)
PIKA_OUTPUT_QUEUE_NAME: str = getenv(
    "PIKA_OUTPUT_QUEUE_NAME", "ic_microservice_output_queue"
)
PIKA_RABBITMQ_HOST: str = getenv("PIKA_RABBITMQ_HOST", "localhost")
CLASSIFY_QUEUE: str = getenv("CLASSIFY_QUEUE", "classify_queue")
VECTORISE_QUEUE: str = getenv("VECTORISE_QUEUE", "vectorise_queue")


class ICMPikaClient(object):
    """Issue Classifier Microservice Pika RabbitMQ Client.

    This is a wrapper class around pika suitable for quickly
    getting the microservice up and running consuming classification requests.

    Default values for several configuration options, such as
    the host for the running RabbitMQ instance, can be found above.
    """

    def __init__(self) -> None:
        """Initialise the pika client.

        This init function performs the following:
        1. Establishes a connection to the RabbitMQ instance.
        2. Declares a RabbitMQ exchange for both classification requests and results.
        3. Declares the input queue for the classification requests.
        4. Declares the output queue for the classification results.
        5. Binds the routing keys to the input and output queues.
        """
        self._init_connection()
        self._declare_exchange()
        self._declare_input_queue()
        self._declare_output_queue()
        self._bind_routing_keys_to_queues()

    def _init_connection(self) -> None:
        """Establish a connection to the RabbitMQ instance.

        Uses the follwing environment variable:
            - PIKA_RABBITMQ_HOST: Hostname of the running RabbitMQ instance to connect to.
        """
        connection = BlockingConnection(ConnectionParameters(host=PIKA_RABBITMQ_HOST))
        self.channel: BlockingChannel = connection.channel()

    def _declare_exchange(self) -> None:
        """Declare a RabbitMQ exchange for both classification requests and results.

        The exchange will be used for forwarding the incoming classification
        requests as well as outgoing classification results to the RabbitMQ
        instance. If an exchange of the same name and type exists, it will be
        reused by the microservice.

        Uses the follwing environment variables:
            - PIKA_EXCHANGE_NAME: The name of the exchange to be declared.
            - PIKA_EXCHANGE_TYPE: The type of the exchange to be declared.
        """
        self.channel.exchange_declare(
            exchange=PIKA_EXCHANGE_NAME, exchange_type=PIKA_EXCHANGE_TYPE
        )

    def _declare_input_queue(self) -> None:
        """Declare the input queue of the classification results.

        The input queue is where all incoming issue classification requests are
        to be passed. If a queue with the same name and method exists, it will
        be reused.

        Uses the following environment variables:
            - PIKA_QUEUE_NAME: The name of the queue to be declared.
            - PIKA_IS_QUEUE_EXCLUSIVE: Whether the queue should be declared as
            exclusive,
            i.e. whether the queue can only be used by the channel of the declaring running pika client.
        """
        input_queue: Any = self.channel.queue_declare(
            queue=PIKA_INPUT_QUEUE_NAME, durable=True
        )
        self.input_queue: Optional[str] = input_queue.method.queue

    def _declare_output_queue(self) -> None:
        """Declare the output queue for the classification results.

        Analogously to the declaration of the input queue, this declares the
        output queue where all outgoing classification results are to be forwarded.

        Uses the follwing environment variables:
            - PIKA_QUEUE_NAME: The name of the queue to be declared.
            - PIKA_IS_QUEUE_EXCLUSIVE: Whether the queue should be declared as  exclusive,
        i.e. whether the queue can only be used by the channel of the declaring     running pika client.
        """
        output_queue = self.channel.queue_declare(
            queue=PIKA_OUTPUT_QUEUE_NAME, durable=True
        )
        self.output_queue: Optional[str] = output_queue.method.queue

    def _bind_routing_keys_to_queues(self) -> None:
        """Bind the routing keys to the input and output queues.

        In order for the single exchange to know where to send the messages to,
        it depends on routing keys. Each message arriving the exchange has a
        routing key attached. Based on that routing key, a message is either
        forwarded to the input or the output queue.

        Uses the following environment variables:
            - PIKA_EXCHANGE_NAME: The name of the exchange to be
            - PIKA_INPUT_ROUTING_KEY: The routing key linking the input queue to
            the exchange.
            - PIKA_OUTPUT_ROUTING_KEY: The routing key linking the output queue
            to the exchange.
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

    def _deserialise_issue_request(
        self,
        message_body: bytes,
    ) -> List[IndexedIssue]:
        """Deserialise an incoming issue classification request.

        The input should have be serialised in JSON consisting of a JSON array
        of JSON objects encoded in either UTF-8, UTF-16 or UTF-32. Each JSON
        object should consist of two key-value pairs: the issue body as a JSON
        string with the key "body" (without quotes), and the issue index as a
        JSON string with the key "id" (without quotes). This id should be unique
        on the client's side (preferably globally irrespective of issues
        currently in the classifier itself) so that the client can map the
        classification results back to their original issues based on that id.

        Parsing is performed in two steps: the input is parsed from bytes into
        JSON using json.loads, then the result is parsed into pydantic
        IndexedIssue objects. Both of these operations are performed by the
        single function call parse_raw_as, which acts as a wrapper for
        json.loads and parse_raw.

        Args: message_body (bytes): The input issue to be classified.

        Returns: List[IndexedIssue]: The list of issues parsed as pydantic
        IndexedIssues.
        """
        logging.info("Deserialising issues...")
        indexed_issues: List[IndexedIssue] = []
        indexed_issues = parse_raw_as(List[IndexedIssue], message_body)
        logging.info("Issues deserialised successfully.")

        return indexed_issues

    def _handle_issue_request(
        self,
        channel: BlockingChannel,
        method_frame: Basic.Deliver,
        header_frame: BasicProperties,
        message_body: bytes,
    ) -> None:
        """Handle an incoming issue classification request.

        Once an incoming classification request is received, it's deserialised
        from JSON into Tuples. Each Tuple contains (1) the issue body, and (2)
        the ID of the issue. Both body and ID must be strings. The ID is chosen
        as a string to allow for different data types that may be used by
        different services.

        Uses the following environment variables:
            - VECTORISE_QUEUE: The queue to which the issues will be first sent
            for the creation of feature vectors.
            - CLASSIFY_QUEUE: The queue to which the feature vectors created
            from the will be sent from the vectoriser to the classifiers.

        Args:
            channel (BlockingChannel): The pika BlockingChannel through which
            the issue came through.
            method_frame (Basic.Deliver): An object containing the delivery tag,
            the redelivered flag, the routing key used to put the message in the
            queue, and the exchange the message was published to.
            header_frame (BasicProperties): A BasicProperties object. This
            encapsulates various attributes of the message, such as the delivery
            mode, the priority, and the content encoding.
            message_body (bytes): The body of the message.
        """
        indexed_issues: List[IndexedIssue] = self._deserialise_issue_request(
            message_body=message_body
        )

        vectorise_issues.signature(
            (indexed_issues,), queue=VECTORISE_QUEUE
        ).apply_async()
        logging.info("Issues sent to Celery for processing.")

    def start_consuming_issue_requests(self) -> None:
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
        logging.info("Now consuming issue classification requests...")
        self.channel.start_consuming()


if __name__ == "__main__":
    pika_client = ICMPikaClient()
    pika_client.start_consuming_issue_requests()
