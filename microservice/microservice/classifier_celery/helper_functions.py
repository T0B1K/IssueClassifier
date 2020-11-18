"""Helper functions for the Celery tasks."""
import logging
from os import getenv
from typing import List, Optional, Tuple

import ujson
from microservice.models.models import VectorisedIssue
from microservice.tree_logic.classifier_tree import ClassifyTree, ClassifyTreeNode
from pika import BlockingConnection, ConnectionParameters
from pika.adapters.blocking_connection import BlockingChannel

# Environment variables used throughout this module
PIKA_EXCHANGE_NAME = getenv("PIKA_EXCHANGE_NAME", "classification")
PIKA_EXCHANGE_TYPE = getenv("PIKA_EXCHANGE_TYPE", "direct")
PIKA_OUTPUT_QUEUE_NAME = getenv("PIKA_OUTPUT_QUEUE_NAME", "issue_classifier_output")
PIKA_RABBITMQ_HOST = getenv("PIKA_RABBITMQ_HOST", "rabbitmq")
PIKA_OUTPUT_ROUTING_KEY = getenv("PIKA_OUTPUT_ROUTING_KEY", "Classification.Results")
CLASSIFY_QUEUE: str = getenv("CLASSIFY_QUEUE", "classify_queue")


def _init_publisher() -> Tuple[BlockingConnection, BlockingChannel]:
    rabbitmq_connection_parameters = ConnectionParameters(host=PIKA_RABBITMQ_HOST)
    rabbitmq_connection = BlockingConnection(rabbitmq_connection_parameters)
    logging.info("Connection to RabbitMQ established. Setting up channel...")
    rabbitmq_channel = rabbitmq_connection.channel()
    logging.info("Channel set up. Declaring exchange...")
    rabbitmq_channel.exchange_declare(
        exchange=PIKA_EXCHANGE_NAME, exchange_type=PIKA_EXCHANGE_TYPE
    )

    return rabbitmq_connection, rabbitmq_channel


def send_results_to_output(results: List[VectorisedIssue]) -> None:
    """Send the classification results back to the output queue at RabbitMQ.

    Note that not the entire issue is returned. Only the classification results
    and the indices of the issues to which the feature vectors correspond. The
    client can then map the classification results back to the originally sent
    issues using the id.

    Uses the following environment variables:
        - PIKA_EXCHANGE_NAME: The name of the RabbitMQ exchange.
        - PIKA_OUTPUT_ROUTING_KEY: The routing key binding the given exchange to
        the output queue.

    Args:
        results (List[VectorisedIssue]): The transformed issues to be sent to RabbitMQ.
    """
    filtered_results = [result.dict(exclude={"body"}) for result in results]
    serialised_results = ujson.dumps(filtered_results).encode("utf-8")

    logging.info("Declaring exchange...")
    rabbitmq_connection, rabbitmq_channel = _init_publisher()
    logging.info("Exchange declared. Sending classifications now...")
    rabbitmq_channel.basic_publish(
        exchange=PIKA_EXCHANGE_NAME,
        routing_key=PIKA_OUTPUT_ROUTING_KEY,
        body=serialised_results,
    )

    logging.info("Classifications sent. Closing connection now...")
    rabbitmq_connection.close()
    logging.info("Connection closed. Goodbye :)")


def get_node(
    node_index: int,
    custom_node: Optional[ClassifyTreeNode],
    classify_tree: ClassifyTree,
) -> ClassifyTreeNode:
    """Get the classifier tree node.

    The node is returned based on the given node index and whether a custom node
    has been supplied (although supplying a custom node is not the recommended
    method. See the documentation for classify_issues).

    Args:
        node_index (int): The index of the node to be returned.
        custom_node (Optional[ClassifyTreeNode]): The custom node supplied.
        classify_tree (ClassifyTree): The classifier tree instance of the worker.

    Returns:
        ClassifyTreeNode: The chosen classifier tree node to be used.
    """
    current_node: Optional[ClassifyTreeNode] = None
    if custom_node is not None:
        current_node = custom_node
    else:
        current_node = classify_tree.get_node(node_index)

    return current_node
