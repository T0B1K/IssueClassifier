from typing import Any, List, Optional, Tuple

from config.classifier_configuration import Configuration
from microservice.ms_celery.celery import app as celery_app
from microservice.ms_celery.classifier_tree import ClassifyTree, ClassifyTreeNode
from numpy import ndarray
from celery import Task
from celery.canvas import group

default_label_classes = Configuration().get_value_from_config("labelClasses")


def ClassifyTask(Task):

    _classify_tree: Optional[ClassifyTree] = None

    def __init__(self, label_classes: List[str] = default_label_classes) -> None:
        if self._classify_tree is None:
            ClassifyTask._classify_tree = ClassifyTree(
                label_classes,
            )

    @property
    def classify_tree(self):
        return self._classify_tree


@celery_app.task(base=ClassifyTask)
def classify_issues(
    issues: List[Tuple[ndarray[Any], List[str], int]], node_index: int = 1
):
    classify_tree: ClassifyTree = classify_issues.classify_tree
    max_node_index = classify_tree.get_node_count()

    current_node: ClassifyTreeNode = classify_tree.get_node(node_index)
    to_left_child, to_right_child = current_node.classify(issues)

    if 2 * node_index > max_node_index:
        # TODO return issues to RabbitMQ
        pass
    else:
        left_child_index = 2 * node_index
        right_child_index = 2 * node_index + 1
        group(
            classify_issues(to_left_child, left_child_index),
            classify_issues(to_right_child, right_child_index),
        )
