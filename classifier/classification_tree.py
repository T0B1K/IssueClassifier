import load_classifier
import numpy
# issiue = tupel[str,list[str],int]


class ClassificationTree:
    """ This class provides a classification tree """

    def __init__(self, labelClasses: numpy.ndarray):
        """Constructor for classification trees.

        Args:
            labelClasses (numpy.ndarray): A list of labels.
        """
        
        self.rootNode = rootNode(labelClasses)

    def classify(self, data: numpy.ndarray) -> list:
        """This method classifies issues.

        Args:
            data (numpy.ndarray): A list of documents.

        Returns:
            list: An ordered List[List[str]] for the given documents.
        """
        if data.size == 0:
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

    def __init__(self, labelClasses: numpy.ndarray, knowledge: numpy.ndarray):
        """Constructor for node objects.

        Args:
            labelClasses (numpy.ndarray): Array of labels as Strings.
            knowledge (numpy.ndarray): Array of checked labels by parent Nodes.
        """
        if not labelClasses.size:
            raise("There are no labelClasses provided")
        if not knowledge.size:
            raise("There is no knowledge being provided")
        self.labelClasses: numpy.ndarray = labelClasses[0]
        self.knowledge: numpy.ndarray = knowledge
        self.vectorizer = load_classifier.getVectorizer()
        self.classifier = load_classifier.getClassifier(
            [self.labelClasses] + self.knowledge)

        if len(labelClasses) == 1:
            self.leftChild = None
            self.rightChild = None
        else:
            self.leftChild: Node = Node(
                labelClasses[1:], self.knowledge + [("{}".format(self.labelClasses))])
            self.rightChild: Node = Node(
                labelClasses[1:], self.knowledge + [("not{}".format(self.labelClasses))])

    def classify(self, data: numpy.ndarray) -> numpy.ndarray:
        """This method classifies issues.

        Args:
            data (numpy.ndarray): list of documents

        Returns:
            numpy.ndarray: Ordered List[List[labels]] for the given documents
        """
        if not data.size:
            raise("No data was provided")
        toleftChild: list = []
        torightChild: list = []
        for issue in data:
            vectorizedIssue: numpy.ndarray = self.vectorizer.transform([
                                                                       issue[0]])
            prediction: numpy.ndarray = self.classifier.predict(
                vectorizedIssue)
            if prediction[0] == 0:
                issue[1].append(self.labelClasses)
                toleftChild.append(issue)
            else:
                torightChild.append(issue)
        if self.rightChild is None or self.rightChild is None:
            return toleftChild + torightChild
        return self.leftChild.classify(toleftChild) + self.rightChild.classify(torightChild)


class rootNode:
    """ This class provides the implemantation of the rootNode for the classification tree """

    def __init__(self, labelClasses: numpy.ndarray):
        """Constructor for rootNode objects

        Args:
            labelClasses (numpy.ndarray): Array of labels as Strings.
        """
        if not labelClasses.size:
            raise("No labelclasses were provided")
        self.labelClasses: numpy.ndarray = labelClasses[0:2]
        self.classifier = load_classifier.getClassifier(self.labelClasses)
        self.vectorizer = load_classifier.getVectorizer()
        self.leftChild: Node = Node(labelClasses[2:], [labelClasses[0]])
        self.rightChild: Node = Node(labelClasses[2:], [labelClasses[1]])

    def classify(self, data: numpy.ndarray) -> numpy.ndarray:
        """Constructor for rootNode objects

        Args:
            data (numpy.ndarray): Array of labels as Strings.

        Returns:
            numpy.ndarray: Classified array of tuples.
        """
        if not data.size:
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
        if self.rightChild is None or self.rightChild is None:
            return toleftChild + torightChild
        return self.leftChild.classify(toleftChild) + self.rightChild.classify(torightChild)
