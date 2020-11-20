"""Custom base task classes for the Celery tasks classify_issues and vectorise_issues.

This module contains the custom base task classes for classify_issues and
vectorise_issues as defined in the module tasks. Using custom base task classes
allows the instantiation and storage of the classifier tree and vectoriser for
each of classify_issues and vectorise_issues respectively.
"""
from typing import List, Optional

from microservice.config.classifier_config import Configuration
from microservice.config.load_classifier import get_vectoriser
from microservice.tree_logic.classifier_tree import ClassifyTree

import logging
from celery import Task

default_label_classes = Configuration().get_value_from_config("labelClasses")


class ClassifyTask(Task):
    """The classifier tree base task for classify_issues.

    Once a worker is spun up, the __init__ function of ClassifyTask is called
    exactly one. As such, during the lifetime of a worker, the same ClassifyTree
    instance is used for each subsequent task call of classify_issues.
    """

    _classify_tree: Optional[ClassifyTree] = None

    def __init__(self, label_classes: List[str] = default_label_classes) -> None:
        """Initialise the classify_issues task class.

        This is where an instance of ClassifyTree is created and kept during the
        lifetime of the worker. Based on the provided label classes, the
        classifier tree is automatically generated. The default value of
        label_classes is found in load_config.json under "labelClasses" (without
        quotes), but a custom set of label classes can be supplied as well. In
        this case, it is the duty of the user to ensure that corresponding
        classifiers exist for the custom label classes.

        Args:
            label_classes (List[str], optional): The label classes to be used
            for the classifiers. Defaults to default_label_classes.
        """
        if self._classify_tree is None:
            self._classify_tree = ClassifyTree(
                label_classes,
            )
            logging.info(
                "Classifier tree initialised for label classes: " + str(label_classes)
            )

    @property
    def classify_tree(self) -> Optional[ClassifyTree]:
        """Getter for the classifier tree.

        Returns:
            ClassifyTree: The classifier tree.
        """
        return self._classify_tree


class VectoriseTask(Task):
    """The vectoriser base task for vectorise_issues.

    Once a worker is spun up, the __init__ function of VectoriseTask is called
    exactly one. As such, during the lifetime of a worker, the same vectoriser
    instance is used for each subsequent task call of vectorise_issues.
    """

    _vectoriser: Optional[ClassifyTree] = None

    def __init__(self) -> None:
        """Initialise the vectorise_issues task class.

        This is where an instance of the vectoriser is created and kept during
        the lifetime of the worker. The vectoriser is loaded with the help of
        the get_vectoriser function in the load_classifier module. A custom
        classifier can also be provided, in which case it is the duty of the
        user to ensure that the vectoriser is found in the proper path as
        defined in the load_config.json file.
        """
        if self._vectoriser is None:
            self._vectoriser = get_vectoriser()

    @property
    def vectoriser(self):
        """Getter for the vectoriser.

        Returns:
            Any: The vectoriser.
        """
        return self._vectoriser
