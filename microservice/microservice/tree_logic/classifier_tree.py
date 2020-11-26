"""Classifier tree and node logic of the microservice.

This class consists primarily of the two classes responsible for the classifier
tree and the classifier tree's nodes and their accompanying logic.
"""
from __future__ import annotations
import logging

import queue
from queue import Queue
from typing import Generator, List, Optional, Tuple, Union

from microservice.config.load_classifier import get_classifier
from microservice.models.models import VectorisedIssue
from numpy import ndarray


class ClassifyTreeNode:
    """The classifier tree node class.

    This class encapsulates the logic behind a single classifier tree node.
    """

    def __init__(
        self,
        label_classes: List[str],
        knowledge: str = "",
        is_root_node: bool = False,
    ) -> None:
        """Initialise a classifier tree node.

        The classifier tree node encapsulates various elements, including (1)
        the label classes with which it works, (2) the knowledge about the
        issues obtained thus far (i.e. what the node knows about the issue, for
        example a given issue represents a bug but not related to API), (3)
        whether it is the root node of the classifier tree, (4) its classifier,
        and (5) its child nodes.

        Args:
            label_classes (List[str], optional): The label class(es) of the
            current node. Defaults to label_classes_from_config.
            knowledge (List[str], optional): The currently obtained knowledge
            about the issues that are given to it. Defaults to [].
            is_root_node (bool, optional): Whether the current node is a root
            node of the classifier tree or not. Defaults to False.

        Raises:
            ValueError: [description]
        """
        if not label_classes:
            raise ValueError("Label classes has not been given as argument")

        self._knowledge: str = knowledge
        self._is_root_node: bool = is_root_node

        self._label_classes: Optional[Union[List[str], str]] = None
        self._right_child: Optional[ClassifyTreeNode] = None
        self._left_child: Optional[ClassifyTreeNode] = None
        self._child: Optional[ClassifyTreeNode] = None

        self._init_current_node_label_classes(label_classes=label_classes)
        self._get_classifier_for_current_node()
        self._init_children(
            label_classes=label_classes,
        )

        logging.debug(
            "Node initialisation successful."
            + "\nCurrent node label classes: "
            + str(self._label_classes)
            + "\nCurrent node knowledge: "
            + str(self._knowledge)
        )

    def _init_current_node_label_classes(self, label_classes: List[str]) -> None:
        if self._is_root_node:
            self._label_classes = label_classes[0:2]
        else:
            self._label_classes = label_classes[0]

    def _get_classifier_for_current_node(self) -> None:
        """Get the classifier of the current classifier tree node.

        If the current node is a root node, the first two labels are used to
        retrieve the classifier (in our original implementation, they represent
        bug and enhancement).

        If the current node is not a root node, the first label along with the
        current node's knowledge (i.e. what the node knows about the issue, for
        example a given issue represents a bug but not related to API) is used
        to retrieve the classifier.

        Args:
            label_classes (List[str]): The input complete label classes from
            which the classifier is to be retrieved.
        """
        if self._is_root_node:
            self._classifier = get_classifier(labels=self._label_classes)  # type: ignore
        else:
            self._classifier = get_classifier(
                labels=[
                    "{}_{}".format(self._label_classes, self._knowledge),
                    self._knowledge,
                ]
            )

    def _init_children(self, label_classes: List[str]) -> None:
        """Initialise the children of the current node.

        If the current node is a root node, then it classifies issues as either
        a bug or enhancement; both labels are mutually exclusive. In this case,
        the remaining children are to cover the remaining issue labels.

        If the current node is not a root node, then it either attaches the
        label it's responsible for (e.g. "api", "doku" (without quotes)) or it
        does not. In addition, the current node can make use of what labels have
        already been attached to the issue(s) it handles.

        Note that this imposes a requirement on how label classes are ordered in
        the configuration.

        Args:
            label_classes (List[str], optional): The label classes of the
            current node.
        """
        if self._is_root_node:
            self._left_child = ClassifyTreeNode(
                label_classes=label_classes[2:],
                knowledge=label_classes[0],
            )
            self._right_child = ClassifyTreeNode(
                label_classes=label_classes[2:],
                knowledge=label_classes[1],
            )
        else:
            if len(label_classes) != 1:
                self._child = ClassifyTreeNode(
                    label_classes=label_classes[1:],
                    knowledge=self._knowledge,
                )

    def has_children(self) -> bool:
        """Return whether the current node has child nodes or not.

        Returns:
            bool: True if the current node has any child nodes, and false
            otherwise (this would imply that the current node is a leaf node).
        """
        return ((self._left_child is not None) and (self._right_child is not None)) or (
            self._child is not None
        )

    def is_root_node(self) -> bool:
        """Return whether the current node is a root node or not.

        Returns:
            bool: True if the current node is a root node, and false otherwise.
        """
        return self._is_root_node

    def get_children(
        self,
    ) -> Union[ClassifyTreeNode, Tuple[ClassifyTreeNode, ClassifyTreeNode]]:
        """Return the children of the current node.

        Raises:
            Exception: When the current node has no children.

        Returns:
            Tuple[ClassifyTreeNode, ClassifyTreeNode]: Tuple where the first and
            second elements correspond to the left and right child nodes, respectively.
        """
        if self.has_children():
            if self._child is not None:
                return self._child
            else:
                return self._left_child, self._right_child  # type: ignore

        else:
            raise Exception("Current node has no children")

    def _determine_input_for_children(
        self,
        prediction: ndarray,
        current_issue: VectorisedIssue,
        to_left_child: List[VectorisedIssue],
        to_right_child: List[VectorisedIssue],
    ) -> Tuple[List[VectorisedIssue], List[VectorisedIssue]]:
        """Determine to which node the given issue with its prediction is to be forwarded to.

        If the forwarding node is the root node, then exactly one label out of
        bug/enhancement has to be attached. Issues with "bug" (without quotes)
        attached as a label are then forwarded to the left child node (which
        makes use of the fact that these issues have not been labelled as
        "enhancement" (without quotes)). The same applies analogously to issues
        with "enhancement" (without quotes) attached as a label to the right
        child node.

        Args:
            prediction (ndarray): The prediction of the current issue
            current_issue (VectorisedIssue): The current issue to be forwarded.
            to_left_child (List[VectorisedIssue]): The list of issues destined
            for the left child node.
            to_right_child (List[VectorisedIssue]): The list of issues destined
            for the right child node.

        Returns:
            Tuple[List[VectorisedIssue], List[VectorisedIssue]]: Tuple
            consisting of two elements, where the first and second elements
            correspond to the list of issues destined for the left child and
            right child nodes, respectively.
        """
        current_issue_labels: List[str] = current_issue.labels

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
                current_issue_labels.append(str(self._label_classes))
                to_left_child.append(current_issue)
            else:
                to_right_child.append(current_issue)

        return to_left_child, to_right_child

    def classify(
        self, issues: List[VectorisedIssue]
    ) -> Tuple[List[VectorisedIssue], List[VectorisedIssue]]:
        """Produce the prediction of the label for the input transformed issues.

        The label that will be attached is based on the specific node that
        performs the classification. After classification, the issues destinated
        to the left child node and the right child node are determined and
        returned.

        Args:
            issues (List[VectorisedIssue]): The list of transformed issues to be
            classified.

        Raises:
            ValueError: If no issues have been passed.

        Returns:
            Tuple[List[VectorisedIssue], List[VectorisedIssue]]: Tuple of two
            lists, where the first and seconds elements correspond to the list
            of issues for the left and right child nodes, respectively.
        """
        if issues is None:
            raise ValueError("Invalid argument for issues!")

        to_left_child: List[VectorisedIssue] = []
        to_right_child: List[VectorisedIssue] = []
        for current_issue in issues:
            current_issue_body: ndarray = current_issue.body
            prediction: ndarray = self._classifier.predict(current_issue_body)
            to_left_child, to_right_child = self._determine_input_for_children(
                prediction,
                current_issue,
                to_left_child,
                to_right_child,
            )

        return to_left_child, to_right_child


class ClassifyTree:
    """The classifier tree class.

    This class encapsulates the logic behind a single classifier tree. The
    classifier tree consists of a single node that is the root node of the
    classifier tree instance.
    """

    def __init__(self, label_classes: List[str]) -> None:
        """Initialise the classifier tree.

        The classifier tree is generated based on the input label classes. The
        default label classes are called from load_config.json with the help of
        the load_classifier module. However, custom label classes can be input
        as well. In such a case, it is the duty of the user to ensure that the
        appropriate classifiers are stored under trained_classifiers in
        pickelised form.

        Args:
            label_classes (List[str], optional): The label classes based on
            which the classifier tree will be generated. Defaults to label_classes_from_config.
        """
        self._root_node = ClassifyTreeNode(
            label_classes=label_classes, is_root_node=True
        )

    def tree_node_generator(
        self,
    ) -> Generator[ClassifyTreeNode]:
        """Return a level-order generator for the classifier tree.

        The generator iterates over the nodes in the classifier tree starting
        from the root node whose index is 1. The iteration is level-order.

        Yields:
            Iterator[ClassifyTreeNode]: The level-order generator of the
            classifier tree.
        """
        self._node_queue: "Queue[ClassifyTreeNode]" = queue.Queue()
        self._node_queue.put(self._root_node)

        while not self._node_queue.empty():
            current_node: ClassifyTreeNode = self._node_queue.get()
            if current_node.has_children():
                if current_node._is_root_node:
                    left_child, right_child = current_node.get_children()  # type: ignore
                    self._node_queue.put(left_child)
                    self._node_queue.put(right_child)
                else:
                    child = current_node.get_children()
                    self._node_queue.put(child)  # type: ignore

            yield current_node

    def get_node_count(self) -> int:
        """Get the number of nodes in the tree.

        Returns:
            int: The number of nodes in the classifier tree.
        """
        count = 0
        for _ in self.tree_node_generator():
            count += 1

        return count

    def get_node(self, index: int) -> ClassifyTreeNode:  # type: ignore
        """Return the node based on the input index.

        Args:
            index (int): The index of the node in the classifier tree.

        Returns:
            ClassifyTreeNode: The node corresponding to the input index
        """
        tree_node_generator = self.tree_node_generator()

        if index <= self.get_node_count():
            for current_node in tree_node_generator:
                index -= 1
                if index == 0:
                    return current_node
