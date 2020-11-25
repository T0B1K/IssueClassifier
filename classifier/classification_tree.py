import load_classifier
import numpy
import json
# issiue = tupel[str,list[str],int]


class ClassificationTree:
    """ This class provides a classification tree """

    def __init__(self, labelClasses: list):
        """Constructor for classification trees.

        Args:
            labelClasses (numpy.ndarray): A list of labels.
        """
        
        self.rootNode = rootNode(labelClasses)

    def classify(self, data: list) -> list:
        """This method classifies issues.

        Args:
            data (numpy.ndarray): A list of documents.

        Returns:
            list: An ordered List[List[str]] for the given documents.
        """
        if not data:
            raise("There are no documents to classify")
        toClassify: list = [(issue, [], idx)for idx, issue in enumerate(data)]
        classified: numpy.ndarray = self.rootNode.classify(toClassify)
        sortedClassified: list = (
            sorted(classified, key=lambda element: element[2]))
        sortedList:list = [issue[1] for issue in sortedClassified]
        assert sortedList, "Sorted list was expected not to be empty"
        return sortedList


class Node:
    """Class provides a node implementation for the ClassificationTree """

    def __init__(self, labelClasses: list, knowledge: list):
        """Constructor for node objects.

        Args:
            labelClasses (numpy.ndarray): Array of labels as Strings.
            knowledge (numpy.ndarray): Array of checked labels by parent Nodes.
        """
        if not labelClasses:
            raise("There are no labelClasses provided")
        if not knowledge:
            raise("There is no knowledge being provided")
        self.labelClasses: numpy.ndarray = labelClasses[0]
        self.knowledge: str = knowledge
        self.vectorizer = load_classifier.getVectorizer()
        self.classifier = load_classifier.getClassifier(["{}_{}".format(self.labelClasses,self.knowledge),self.knowledge])

        if len(labelClasses) == 1:
            self.child = None
        else:
            self.child: Node = Node(
                labelClasses[1:], self.knowledge)

    def classify(self, data: list) -> list:
        """This method classifies issues.

        Args:
            data (numpy.ndarray): list of documents

        Returns:
            numpy.ndarray: Ordered List[List[labels]] for the given documents
        """
        if not data:
            raise("No data was provided")
        for issue in data:
            vectorizedIssue: numpy.ndarray = self.vectorizer.transform([
                                                                       issue[0]])
            prediction: numpy.ndarray = self.classifier.predict(
                vectorizedIssue)
            if prediction[0] == 0:
                issue[1].append(self.labelClasses)
                
        if self.child is None:
            return data
        else:
            return self.child.classify(data)

class rootNode:
    """ This class provides the implemantation of the rootNode for the classification tree """

    def __init__(self, labelClasses: list):
        """Constructor for rootNode objects

        Args:
            labelClasses (numpy.ndarray): Array of labels as Strings.
        """
        if not labelClasses:
            raise("No labelclasses were provided")
        self.labelClasses: numpy.ndarray = labelClasses[0:2]
        self.classifier = load_classifier.getClassifier(self.labelClasses)
        self.vectorizer = load_classifier.getVectorizer()
        self.leftChild: Node = Node(labelClasses[2:], labelClasses[0])
        print("LeftChild_{}_{}".format(labelClasses[2:], labelClasses[0]))
        self.rightChild: Node = Node(labelClasses[2:], labelClasses[1])
        print("RightChild_{}_{}".format(labelClasses[2:], labelClasses[1]))

    def classify(self, data: list) -> list:
        """Constructor for rootNode objects

        Args:
            data (numpy.ndarray): Array of labels as Strings.

        Returns:
            numpy.ndarray: Classified array of tuples.
        """
        if not data:
            raise("No data was provided")
        toleftChild: list = []
        torightChild: list = []
        for issue in data:
            vectorizedIssue: numpy.ndarray = self.vectorizer.transform([
                                                                       issue[0]])
            prediction: numpy.ndarray = self.classifier.predict(
                vectorizedIssue)
            if prediction[0] == 0:
                issue[1].append(self.labelClasses[0])
                toleftChild.append(issue)
            else:
                issue[1].append(self.labelClasses[1])
                torightChild.append(issue)

        if self.leftChild is None or self.rightChild is None:
            return toleftChild + torightChild

        if not toleftChild and  torightChild:
            return self.rightChild.classify(torightChild)

        elif not torightChild and toleftChild:
            return self.leftChild.classify(toleftChild)

        return self.leftChild.classify(toleftChild) + self.rightChild.classify(torightChild)
