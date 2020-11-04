from __future__ import annotations

import queue
from queue import Queue
from typing import Any, List, Optional, Tuple

from microservice.config.classifier_configuration import Configuration
from microservice.config.load_classifier import get_classifier
from numpy import ndarray

label_classes_from_config: List[str] = Configuration().get_value_from_config(
    "labelClasses"
)


class ClassifyTreeNode:
    def __init__(
        self,
        label_classes: List[str] = label_classes_from_config,
        knowledge: List[str] = [],
        is_root_node: bool = False,
    ) -> None:
        if not label_classes:
            raise ValueError("Label classes has not been given as argument")

        self._knowledge: List[str] = knowledge
        self._is_root_node: bool = is_root_node

        self._right_child: Optional[ClassifyTreeNode] = None
        self._left_child: Optional[ClassifyTreeNode] = None

        self._init_children(
            label_classes=label_classes, is_root_node=self._is_root_node
        )

    def _init_children(
        self, label_classes: List[str], is_root_node: bool = False
    ) -> None:
        if self._is_root_node:
            self._label_classes = label_classes[0:2]
            self._classifier = get_classifier(categories=self._label_classes)

            self._right_child = ClassifyTreeNode(
                label_classes=label_classes[2:],
                knowledge=[label_classes[1]],
            )
            self._left_child = ClassifyTreeNode(
                label_classes=label_classes[2:],
                knowledge=[label_classes[0]],
            )
        else:
            self._label_classes = label_classes[0]

            if len(label_classes) != 1:
                self._classifier = get_classifier(
                    categories=[self._label_classes] + self._knowledge
                )
                self._right_child = ClassifyTreeNode(
                    label_classes=label_classes[1:],
                    knowledge=self._knowledge + [("{}".format(self._label_classes))],
                )
                self._left_child = ClassifyTreeNode(
                    label_classes=label_classes[1:],
                    knowledge=self._knowledge + [("not{}".format(self._label_classes))],
                )

    def has_children(self) -> bool:
        return (self._left_child is not None) and (self._right_child is not None)

    def get_children(self) -> Tuple[ClassifyTreeNode, ClassifyTreeNode]:
        if self.has_children():
            return self._left_child, self._right_child  # type: ignore
        else:
            raise Exception("Current node has no children")

    def _determine_input_for_children(
        self,
        prediction: ndarray[Any],
        current_issue: Tuple[ndarray[Any], List[str], int],
        to_left_child: List[Tuple[ndarray[Any], List[str], int]],
        to_right_child: List[Tuple[ndarray[Any], List[str], int]],
    ) -> Tuple[
        List[Tuple[ndarray[Any], List[str], int]],
        List[Tuple[ndarray[Any], List[str], int]],
    ]:
        current_issue_labels: List[str] = current_issue[1]

        if self._is_root_node:
            if prediction[0] == 0:
                current_issue_labels.append(self._label_classes[0])
                to_left_child.append(current_issue)
            else:
                current_issue_labels.append(self._label_classes[1])
                to_right_child.append(current_issue)
            pass
        else:
            if prediction[0] == 0:
                current_issue_labels.append(self._label_classes[0])
                to_left_child.append(current_issue)
            else:
                to_right_child.append(current_issue)

        return to_left_child, to_right_child

    def classify(
        self, issues: List[Tuple[ndarray[Any], List[str], int]]
    ) -> Tuple[
        List[Tuple[ndarray[Any], List[str], int]],
        List[Tuple[ndarray[Any], List[str], int]],
    ]:
        if issues is None:
            raise ValueError("Invalid argument for issues!")

        to_left_child: List[Tuple[ndarray[Any], List[str], int]] = []
        to_right_child: List[Tuple[ndarray[Any], List[str], int]] = []
        for current_issue in issues:
            prediction: ndarray[Any] = self._classifier.predict(current_issue)
            to_left_child, to_right_child = self._determine_input_for_children(
                prediction,
                current_issue,
                to_left_child,
                to_right_child,
            )

        return to_left_child, to_right_child


class ClassifyTree:
    def __init__(self, label_classes: List[str] = label_classes_from_config) -> None:
        self._root_node = ClassifyTreeNode(
            label_classes=label_classes, knowledge=[], is_root_node=True
        )

    def tree_node_generator(
        self,
    ):
        self._node_queue: "Queue[ClassifyTreeNode]" = queue.Queue()
        self._node_queue.put(self._root_node)

        if not self._node_queue.empty():
            current_node: ClassifyTreeNode = self._node_queue.get()
            if current_node.has_children():
                left_child, right_child = current_node.get_children()
                self._node_queue.put(left_child)
                self._node_queue.put(right_child)
            yield current_node

    def get_node_count(self) -> int:
        count = 0
        for node in self.tree_node_generator():
            count += 1
        return count

    def get_node(self, index) -> ClassifyTreeNode:  # type: ignore
        tree_node_generator = self.tree_node_generator()

        if index <= self.get_node_count():
            for current_node in tree_node_generator:
                index -= 1
                if index == 0:
                    return current_node
