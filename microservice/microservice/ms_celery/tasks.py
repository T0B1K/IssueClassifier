import os
from typing import Any, List, Optional, Tuple, Union

from microservice.config.classifier_configuration import Configuration
from microservice.config.load_classifier import get_classifier, get_vectoriser
from microservice.ms_celery.celery import app as celery_app
from numpy import ndarray

from celery.app.task import Task
from celery.canvas import group

classifier_config = Configuration()
label_classes_from_config = classifier_config.get_value_from_config("labelClasses")
CLASSIFY_QUEUE: str = os.getenv("CLASSIFY_QUEUE", "classify_queue")


class ClassifyNodeTask(Task):

    name = "tasks.ClassifyNodeTask"

    _vectoriser: Any = None

    def __init__(
        self,
        label_classes: List[str] = label_classes_from_config,
        knowledge: List[str] = [],
    ) -> None:
        if not label_classes:
            try:
                label_classes = label_classes_from_config
            except Exception:
                raise ValueError("Label classes has not been given as argument")

        self._knowledge: List[str] = knowledge
        self._is_root_node: bool = self._knowledge == []

        ClassifyNodeTask._init_vectoriser()
        self._vectoriser: Any = ClassifyNodeTask._vectoriser

        self._right_child: Optional[ClassifyNodeTask] = None
        self._left_child: Optional[ClassifyNodeTask] = None
        if self._is_root_node:
            self._label_classes = label_classes[0:2]
            self._classifier = get_classifier(categories=self._label_classes)

            self._right_child = ClassifyNodeTask(
                label_classes=label_classes[2:],
                knowledge=[label_classes[1]],
            )
            self._left_child = ClassifyNodeTask(
                label_classes=label_classes[2:],
                knowledge=[label_classes[0]],
            )
        else:
            self._label_classes = label_classes[0]

            if len(label_classes) != 1:
                self._classifier = get_classifier(
                    categories=[self._label_classes] + self._knowledge
                )
                self._right_child = ClassifyNodeTask(
                    label_classes=label_classes[1:],
                    knowledge=self._knowledge + [("{}".format(self._label_classes))],
                )
                self._left_child = ClassifyNodeTask(
                    label_classes=label_classes[1:],
                    knowledge=self._knowledge + [("not{}".format(self._label_classes))],
                )

    def run(
        self,
        issues: Union[List[str], List[Tuple[str, List[str], int]]],
        *args,
        **kwargs
    ):
        if issues is None or issues == []:
            raise ValueError(
                "A valid argument for issues to be classified must be passed!"
            )

        indexed_issues: List[Tuple[str, List[str], int]]
        if self._is_root_node:
            indexed_issues = [(issue, [], index) for index, issue in enumerate(issues)]  # type: ignore
        else:
            indexed_issues = issues  # type: ignore

        to_left_child: List[Tuple[str, List[str], int]] = []
        to_right_child: List[Tuple[str, List[str], int]] = []
        for current_issue in indexed_issues:
            current_issue_body = current_issue[0]
            current_issue_labels = current_issue[1]
            feature_vectors: ndarray = self._vectoriser.transform([current_issue_body])
            prediction: ndarray = self._classifier.predict(feature_vectors)
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
        if self._left_child is None or self._right_child is None:
            return to_left_child + to_right_child
        else:
            left_child_sig = self._left_child.signature(
                args=(to_left_child,), queue=CLASSIFY_QUEUE
            )
            right_child_sig = self._right_child.signature(
                args=(to_right_child,), queue=CLASSIFY_QUEUE
            )

            group(left_child_sig, right_child_sig).apply_async()

    @classmethod
    def _init_vectoriser(cls) -> None:
        if cls._vectoriser is None:
            cls._vectoriser = get_vectoriser()


celery_app.tasks.register(ClassifyNodeTask())  # type: ignore
