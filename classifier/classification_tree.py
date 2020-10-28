import load_classifier
import numpy
# issiue = tupel[str,list[str],int] 
class ClassificationTree:
    """ This class provides a classification tree """
    def __init__(self,labelClasses: numpy.ndarray ):
        """
        Description: Constructor for classification trees
        Input: List[String] of labels
        Output: ClassificationTree object
        """
        self.rootNode = rootNode(labelClasses)
        
         
    
    def classify(self,data: numpy.ndarray):
        """Description: This method classifies issues
        Input: List[String] of documents
        Output: Ordered List[List[labels]] for the given documents
        """
        toClassify = [(issue,[],idx)for idx , issue in enumerate(data)]
        classified =  self.rootNode.classify(toClassify)   
        sortedClassified = (sorted(classified, key = lambda element: element[2]))
        filtheredIssiues = [issue[1] for issue in sortedClassified]
        return filtheredIssiues
        
     
    

class Node:
    """ Class provides a node implementation for the ClassificationTree """

    def __init__(self,labelClasses: numpy.ndarray,knowledge: numpy.ndarray):
        """
        Description: Constructor for node objects
        Input:  labelClasses: array of labels as Strings
                knowledge: array of checked labels by parent Nodes
        Output: Returns Node object
        """
        self.labelClasses = labelClasses[0]
        self.knowledge = knowledge
        self.vectorizer = load_classifier.getVectorizer()
        self.classifier = load_classifier.getClassifier([self.labelClasses] + self.knowledge)

        if len(labelClasses) == 1:
            self.leftChild = None
            self.rightChild = None
        else:
            self.leftChild = Node (labelClasses[1:],self.knowledge + [("{}".format(self.labelClasses))])
            self.rightChild = Node (labelClasses[1:],self.knowledge + [("not{}".format(self.labelClasses))])


    def classify (self,data: numpy.ndarray)-> numpy.ndarray:
        """
        Description: This method classifies issues
        Input: List[String] of documents
        Output: Ordered List[List[labels]] for the given documents
        """
        toleftChild = []
        torightChild = []
        for issue in data:
            vectorizedIssue = self.vectorizer.transform([issue[0]])
            prediction = self.classifier.predict(vectorizedIssue)
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
        """
        Description: Constructor for rootNode objects
        Input:  labelClasses: array of labels as Strings
        Output: rootNode Object
        """
        self.labelClasses = labelClasses[0:2]
        self.classifier = load_classifier.getClassifier(self.labelClasses)
        self.vectorizer = load_classifier.getVectorizer()
        self.leftChild = Node(labelClasses[2:],[labelClasses[0]])
        self.rightChild = Node(labelClasses[2:],[labelClasses[1]])
    
    def classify (self,data: numpy.ndarray)-> numpy.ndarray:
        """
        Description: Constructor for rootNode objects
        Input:  labelClasses: array of labels as Strings
        Output: rootNode Object
        """
        toleftChild = []
        torightChild = []
        for issue in data:
            vectorizedIssue = self.vectorizer.transform([issue[0]])
            prediction = self.classifier.predict(vectorizedIssue)
            if prediction[0] == 0:
                issue[1].append(self.labelClasses[0])
                toleftChild.append(issue)
            else:
                issue[1].append(self.labelClasses[1])
                torightChild.append(issue)
        if self.rightChild is None or self.rightChild is None:
            return toleftChild + torightChild
        return self.leftChild.classify(toleftChild) + self.rightChild.classify(torightChild)
            
print("Starting...")
tree = ClassificationTree(["bug","enhancement","api","doku"])
print("Result: {}".format(tree.classify(["Howdy,\n\nIt would be great if we could have centered content in the navbar built into Bootstrap.\n\nI find it useful to be able to slightly off-center content depending on the rest of the content in the navbar eg lots of content using .navbar-right... use .navbar-center-left to slightly move it to the left.\n\nI use this to do it which works great:\n\n.navbar-center {\n    position:   absolute;\n    left:       50%;\n    -webkit-transform:  translateX(-50%);\n    -moz-transform: translateX(-50%);\n    -ms-transform:      translateX(-50%);\n    -o-transform:       translateX(-50%);\n    transform:      translateX(-50%);\n}\n\n.navbar-center-left {\n    position:   absolute;\n    left:       40%;\n    -webkit-transform:  translateX(-50%);\n    -moz-transform: translateX(-50%);\n    -ms-transform:      translateX(-50%);\n    -o-transform:       translateX(-50%);\n    transform:      translateX(-50%);\n}\n\n.navbar-center-right {\n    position:   absolute;\n    right:      40%;\n    -webkit-transform:  translateX(-50%);\n    -moz-transform: translateX(-50%);\n    -ms-transform:      translateX(-50%);\n    -o-transform:       translateX(-50%);\n    transform:      translateX(-50%);\n}\n\nThe 40% could easily be a variable for custom aligning per site.\n\nHope you like my idea. :-)\n"])))
print("Ending.....")


