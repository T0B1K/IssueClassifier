from typing import List, Optional

from microservice.config.load_classifier import get_vectoriser
from microservice.config.classifier_configuration import Configuration
from microservice.ms_celery.classifier_tree import ClassifyTree

from celery import Task

default_label_classes = Configuration().get_value_from_config("labelClasses")


class ClassifyTask(Task):

    _classify_tree: Optional[ClassifyTree] = None

    def __init__(self, label_classes: List[str] = default_label_classes) -> None:
        if self._classify_tree is None:
            self._classify_tree = ClassifyTree(
                label_classes,
            )

    @property
    def classify_tree(self):
        return self._classify_tree


class VectoriseTask(Task):

    _vectoriser: Optional[ClassifyTree] = None

    def __init__(self, label_classes: List[str] = default_label_classes) -> None:
        if self._vectoriser is None:
            self._vectoriser = get_vectoriser()

    @property
    def vectoriser(self):
        return self._vectoriser
