"""The tasks module with all the Celery tasks of the microservice.

This module contains two tasks: classify_issues for classification of
VectorisedIssue instances, and vectorise_issues for transformation of
IndexedIssue instances (more precisely, their corresponding body attributes)
into their feature vectors, on which basis classification takes place.

Both tasks inherit from base classes, namely ClassifyTask and VectoriseTask.
This allows for maintaining a state of a task during runtime, in particular the
ClassifyTree for classify_issues and vectoriser for vectorise_issues. As per
Celery logic, the __init__ function of each task is executed only once for each
worker, therefore the same instantiated classifiers and vectorisers are reused
by each worker for each task call.
"""
import logging
from os import getenv
from typing import List
from microservice.classifier_celery.celery import app as celery_app
from microservice.classifier_celery.helper_functions import (
    get_node,
    send_results_to_output,
    determine_issues_per_worker,
)
from microservice.classifier_celery.task_classes import ClassifyTask, VectoriseTask
from microservice.models.models import IndexedIssue, VectorisedIssue
from microservice.tree_logic.classifier_tree import ClassifyTree

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

CLASSIFY_QUEUE: str = getenv("CLASSIFY_QUEUE", "classify_queue")


def _forward_issues(
    node_index: int,
    is_root_node: bool,
    is_leaf_node: bool,
    to_left_child: List[VectorisedIssue],
    to_right_child: List[VectorisedIssue],
) -> None:
    """Forward the issues for further processing or to RabbitMQ back to the client.

    Depending on whether the forwarding node is a leaf node or not, the results
    are forwarded either to the child nodes, or back to RabbitMQ.

    Args:
        node_index (int): The index of the forwarding node in the tree.
        is_leaf_node (bool): Whether the forwarding node is a leaf node.
        to_left_child (List[VectorisedIssue]): The list of VectorisedIssue
        instances to be forwarded to the left child node (if such a node exists.)
        to_right_child (List[VectorisedIssue]): The list of VectorisedIssue
        instances to be forwarded to the right child node (if such a node exists.)
    """
    if is_leaf_node:
        logging.info(
            "Current node is a leaf node. Sending results back to output queue."
        )
        aggregated_results: List[VectorisedIssue] = to_left_child + to_right_child
        if aggregated_results:
            send_results_to_output(aggregated_results)
        else:
            logging.info("Results are empty. Nothing to send, nothing more to do...")
    else:
        logging.info(
            "Current node is not a leaf node. Sending results to corresponding nodes."
        )

        if is_root_node:
            logging.debug("Current node is the root node.")

            left_child_index: int = 2 * node_index
            right_child_index: int = 2 * node_index + 1
            logging.debug("Left child index: " + str(left_child_index))
            logging.debug("Right child index: " + str(right_child_index))

            logging.debug("Sending issues to children now...")
            if to_left_child:
                classify_issues.signature(
                    (to_left_child, left_child_index),
                    queue=CLASSIFY_QUEUE,
                ).delay()

            if to_right_child:
                classify_issues.signature(
                    (to_right_child, right_child_index),
                    queue=CLASSIFY_QUEUE,
                ).delay()
        else:
            logging.debug("Current node is NOT the root node.")
            child_index: int = 2 + node_index
            to_child: List[VectorisedIssue] = to_left_child + to_right_child

            if to_child:
                logging.debug("Sending issue to single child now...")
                classify_issues.signature(
                    (to_child, child_index), queue=CLASSIFY_QUEUE
                ).delay()


@celery_app.task(base=ClassifyTask)
def classify_issues(issues: List[VectorisedIssue], node_index: int = 1) -> None:
    """Classify the issues based on its feature vectors produced by the vectoriser.

    This function outputs a prediction for the input issue based on its body,
    which corresponds to the feature vectors of the original issue body string
    with the same id.

    After outputting a prediction, the worker determines which classifiers
    should further process this issue by splitting the issues into two lists.
    Each list goes to a specific classifier within the tree logic.

    If the worker was running one of the classifiers found as leaf nodes in the
    classifier tree, this signals the end of the classification for these
    issues. the resulting lists are aggregated and sent back to RabbitMQ.

    Note that the same worker can handle multiple classifiers, since each worker
    owns their own instance of each classifier found in the trained_classifiers
    folder, thus each worker possesses the entire classifier tree. Which this
    leads to an increased demand on space, the benefit is that every worker is
    capable of utilising any classifier. Furthermore, should a worker crash, the
    task is retried using another worker, which, under the assumption that every
    worker possesses its own ClassifyTree instance, is possible.

    In addition, the classify_issues task is set to a custom route, i.e.
    classify_issue tasks are routed to a specific queue as defined in
    celery_config.py. This allows for dedicated workers for classification are
    used for classification. Manual routing for each single issue to a specific
    queue is also possible, though this may prove itself overkill for most
    applications.

    Args:
        issues (List[VectorisedIssue]): The list of VectorisedIssue to be classified.
        node_index (int, optional): Index of classifier to be utilised for this
        specific call. If none is specified, the root node (with index 1) is assumed. Defaults to 1.
    """
    logging.info("Current node index: " + str(node_index))
    logging.info("Received issue for classification: " + str(issues))

    classify_tree: ClassifyTree = classify_issues.classify_tree
    max_node_index = classify_tree.get_node_count()
    logging.info("Node count in classification tree: " + str(max_node_index))

    current_node = get_node(node_index=node_index, classify_tree=classify_tree)

    to_left_child: List[VectorisedIssue]
    to_right_child: List[VectorisedIssue]
    to_left_child, to_right_child = current_node.classify(issues)
    logging.info("Issues destined for the left node: " + str(to_left_child))
    logging.info("Issues destined for the right node: " + str(to_right_child))

    is_leaf_node: bool = not current_node.has_children()
    is_root_node: bool = current_node.is_root_node()
    _forward_issues(
        node_index=node_index,
        is_root_node=is_root_node,
        is_leaf_node=is_leaf_node,
        to_left_child=to_left_child,
        to_right_child=to_right_child,
    )


def _forward_issues_to_classifiers(vectorised_issues: List[VectorisedIssue]) -> None:
    issues_per_task: int = determine_issues_per_worker(vectorised_issues)
    chunks: List[List[VectorisedIssue]] = [
        vectorised_issues[x : x + issues_per_task]
        for x in range(0, len(vectorised_issues), issues_per_task)
    ]
    for chunk in chunks:
        classify_issues.signature((chunk,), queue=CLASSIFY_QUEUE).delay()


@celery_app.task(base=VectoriseTask)
def vectorise_issues(
    issues: List[IndexedIssue],
) -> None:
    """Vectorise the input issues.

    Based on the body of each issue, feature vectors are produced. These feature
    vectors are then used by the classifiers to produce a prediction of the
    issue label(s) most suitable for that given issue.

    Since transformation only needs to take place once, an already transformed
    issue will not be transfromed again.

    In addition, the vectorise_issues task is set to a custom route, i.e.
    vectorise_issues tasks are routed to a specific queue as defined in
    celery_config.py. This allows for dedicated workers for transformation.
    Manual routing for each single task to a specific queue is also possible,
    though this may prove itself overkill for most applications.

    Args:
        issues (List[IndexedIssue]): The list of IndexedIssue to be transformed.

    Returns:
        List[VectorisedIssue]: The transformed issues as as list of VectorisedIssue.
    """
    vectorised_issues: List[VectorisedIssue] = []
    vectoriser = vectorise_issues.vectoriser

    for current_issue in issues:
        logging.info("Current issue to be transformed: " + str(current_issue))
        current_issue_body = current_issue.body
        vectorised_current_issue_body = vectoriser.transform([current_issue_body])
        logging.debug("Transformed issue body: " + str(vectorised_current_issue_body))

        vectorised_issue: VectorisedIssue = VectorisedIssue(
            body=vectorised_current_issue_body,
            index=current_issue.index,
            labels=current_issue.labels,
        )
        logging.debug("Transformed issue: " + str(vectorised_issue))

        vectorised_issues.append(vectorised_issue)

    _forward_issues_to_classifiers(vectorised_issues=vectorised_issues)
