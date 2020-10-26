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
        classified =  self.rootNode.classify(data)
        sortedClassified = (sorted(classified, key = lambda element: element[2])) 
        return sortedClassified
        
     
    

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
        self.classifier = None #load_classifier.getClassifier(labelClasses.extend(knowledge))
        self.knowledge = knowledge
        print("Label Classes: {}".format(self.labelClasses))
        print("Knowledge: {}".format(self.knowledge))

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
            prediction = self.classifier.predict(issue[0])
            if prediction[0] == 0:
                issue[1].append(self.labelClasses[0])
                toleftChild.append(issue)
            else:
                torightChild.append(issue)
        if self.rightChild is None or self.rightChild is None:
            return numpy.concatenate((toleftChild,torightChild,),axis= 0)
        return numpy.concatenate((self.leftChild.classify(toleftChild),self.rightChild.classify(torightChild)),axis= 0)

class rootNode:
    """ This class provides the implemantation of the rootNode for the classification tree """
    def __init__(self, labelClasses: numpy.ndarray):
        """
        Description: Constructor for rootNode objects
        Input:  labelClasses: array of labels as Strings
        Output: rootNode Object
        """
        self.labelClasses = labelClasses[0:2]
        self.classifier = None #load_classifier.getClassifier(labelClasses)
        print(self.labelClasses)
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
            prediction = self.classifier.predict(issue[0])
            if prediction[0] == 0:
                issue[1].append(self.labelClasses[0])
                toleftChild.append(issue)
            else:
                issue[1].append(self.labelClasses[1])
                torightChild.append(issue)
        if self.rightChild is None or self.rightChild is None:
            return numpy.concatenate((toleftChild,torightChild,),axis= 0)
        return numpy.concatenate((self.leftChild.classify(toleftChild),self.rightChild.classify(torightChild)),axis = 0)
            
print("Starting...")
tree = ClassificationTree(["bug","enhancement","api","doku"])
print("Ending.....")


