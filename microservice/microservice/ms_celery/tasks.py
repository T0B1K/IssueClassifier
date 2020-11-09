import json
import logging
from os import getenv
from typing import List, Optional, Tuple

from microservice.ms_celery.celery import app as celery_app
from microservice.ms_celery.classifier_tree import ClassifyTree, ClassifyTreeNode
from microservice.ms_celery.task_classes import ClassifyTask, VectoriseTask
from numpy import ndarray
from pika import BlockingConnection, ConnectionParameters

from celery.canvas import group

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

PIKA_EXCHANGE_NAME = getenv("PIKA_EXCHANGE_NAME", "classification")
PIKA_EXCHANGE_TYPE = getenv("PIKA_EXCHANGE_TYPE", "direct")
PIKA_OUTPUT_QUEUE_NAME = getenv("PIKA_OUTPUT_QUEUE_NAME", "issue_classifier_output")
PIKA_RABBITMQ_HOST = getenv("PIKA_RABBITMQ_HOST", "rabbitmq")
PIKA_OUTPUT_ROUTING_KEY = getenv("PIKA_OUTPUT_ROUTING_KEY", "Classification.Results")
CLASSIFY_QUEUE: str = getenv("CLASSIFY_QUEUE", "classify_queue")


def send_results_to_output(results: List[Tuple[ndarray, List[str], int]]):
    rabbitmq_connection_parameters = ConnectionParameters(host=PIKA_RABBITMQ_HOST)

    rabbitmq_connection = BlockingConnection(rabbitmq_connection_parameters)
    logging.info("Connection to RabbitMQ established. Setting up channel...")
    rabbitmq_channel = rabbitmq_connection.channel()
    logging.info("Channel set up. Declaring exchange...")
    rabbitmq_channel.exchange_declare(
        exchange=PIKA_EXCHANGE_NAME, exchange_type=PIKA_EXCHANGE_TYPE
    )

    filtered_results = [result[1:] for result in results]

    logging.info("Exchange declared. Sending classifications now...")
    rabbitmq_channel.basic_publish(
        exchange=PIKA_EXCHANGE_NAME,
        routing_key=PIKA_OUTPUT_ROUTING_KEY,
        body=json.dumps(list(filtered_results)).encode("utf-8"),
    )
    logging.info("Classifications sent. Closing connection now...")
    rabbitmq_connection.close()


@celery_app.task(base=ClassifyTask)
def classify_issues(
    issues: List[Tuple[ndarray, List[str], int]],
    node_index: int = 1,
    custom_node: ClassifyTreeNode = None,
):
    logging.info("Current node index: " + str(node_index))
    logging.info("Received issue for classification: " + str(issues))

    classify_tree: ClassifyTree = classify_issues.classify_tree
    max_node_index = classify_tree.get_node_count()
    logging.info("Node count in classification tree: " + str(max_node_index))

    current_node: Optional[ClassifyTreeNode] = None
    if custom_node is not None:
        current_node = custom_node
    else:
        current_node = classify_tree.get_node(node_index)

    to_left_child: List[Tuple[ndarray, List[str], int]]
    to_right_child: List[Tuple[ndarray, List[str], int]]
    to_left_child, to_right_child = current_node.classify(issues)
    logging.info("Issues destined for the left node: " + str(to_left_child))
    logging.info("Issues destined for the right node: " + str(to_right_child))

    is_leaf_node = not current_node.has_children()
    if is_leaf_node:
        logging.info(
            "Current node is a leaf node. Sending results back to output queue."
        )
        aggregated_results: List[Tuple[ndarray, List[str], int]] = (
            to_left_child + to_right_child
        )
        aggregated_results_is_empty = aggregated_results == []
        if aggregated_results_is_empty:
            logging.info("Results are empty. Nothing to send, nothing more to do...")
        else:
            send_results_to_output(aggregated_results)
    else:
        logging.info(
            "Current node is not a leaf node. Sending results to correspnding nodes."
        )
        left_child_index = 2 * node_index
        right_child_index = 2 * node_index + 1
        logging.info("Left child index: " + str(left_child_index))
        logging.info("Right child index: " + str(right_child_index))
        # custom_left_child, custom_right_child = current_node.get_children()
        # assert (custom_left_child is not None) and (custom_right_child is not None)

        custom_left_child, custom_right_child = None, None

        group(
            classify_issues.signature(
                (to_left_child, left_child_index, custom_left_child),
                queue=CLASSIFY_QUEUE,
            ),
            classify_issues.signature(
                (to_right_child, right_child_index, custom_right_child),
                queue=CLASSIFY_QUEUE,
            ),
        )()


@celery_app.task(base=VectoriseTask)
def vectorise_issues(issues: List[Tuple[ndarray, List[str], int]]):
    vectorised_issues = []
    for current_issue in issues:
        logging.info("Current issue to be transformed: " + str(current_issue))
        current_issue_body = current_issue[0]
        vectorised_current_issue_body = vectorise_issues.vectoriser.transform(
            [current_issue_body]
        )
        logging.info("Transformed issue body: " + current_issue_body)
        current_issue = (
            vectorised_current_issue_body,
            current_issue[1],
            current_issue[2],
        )
        logging.info("Transformed issue: " + str(current_issue))
        vectorised_issues.append(current_issue)

    return vectorised_issues
