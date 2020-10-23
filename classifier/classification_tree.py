import load_classifier
import numpy

class ClassificationTree:
    """ This class provides a classification tree """
    def __init__(self,labelClasses):
        """
        Description: Constructor for classification trees
        Input: List[String] of labels
        Output: ClassificationTree object
        """
        self.rootNode = rootNode(labelClasses)
        
         
    
    def classify(self,data):
        """Description: This method classifies issues
        Input: List[String] of documents
        Output: Ordered List[List[labels]] for the given documents
        """
        classified =  self.rootNode.classify(data)
        sortedClassified = (sorted(classified, key = lambda element: element[2])) 
        return sortedClassified
        
     
    

class Node:
    """ Class provides a node implementation for the ClassificationTree """

    def __init__(self,labelClasses,knowledge):
        """
        Description: Constructor for node objects
        Input:  labelClasses: array of labels as Strings
                knowledge: array of checked labels by parent Nodes
        Output: Returns Node object
        """
        self.labelClasses = labelClasses[0]
        self.classifier = load_classifier.getClassifier(labelClasses.extend(knowledge))
        self.knowledge = knowledge

        if not labelClasses:
            self.leftChild = None
            self.rightChild = None
        else:
            self.leftChild = Node (labelClasses[1:],knowledge.append("{}".format(self.labelClasses)))
            self.rightChild =Node (labelClasses[1:],knowledge.append("not{}".format(self.labelClasses)))


    def classify (self,data):
        """
        Description: This method classifies issues
        Input: List[String] of documents
        Output: Ordered List[List[labels]] for the given documents
        """
        toleftChild = []
        torightChild = []
        for issue in data:
            prediction = self.classifier.predict(issue[0])
            if prediction[0] is 0:
                issue[1].append(self.labelClasses[0])
                toleftChild.append(issue)
            else:
                torightChild.append(issue)
        if self.rightChild is None or self.rightChild is None:
            return numpy.concatenate((toleftChild,torightChild,),axis= 0)
        return numpy.concatenate((self.leftChild.classify(toleftChild),self.rightChild.classify(torightChild)),axis= 0)

class rootNode:
    """ This class provides the implemantation of the rootNode for the classification tree """
    def __init__(self, labelClasses):
        """
        Description: Constructor for rootNode objects
        Input:  labelClasses: array of labels as Strings
        Output: rootNode Object
        """
        self.labelClasses = labelClasses[0:2]
        self.classifier = load_classifier.getClassifier(labelClasses)
        self.leftChild = Node(labelClasses[2:],labelClasses[0])
        self.rightChild = Node(labelClasses[2:],labelClasses[1])
    
    def classify (self,data):
        """
        Description: Constructor for rootNode objects
        Input:  labelClasses: array of labels as Strings
        Output: rootNode Object
        """
        toleftChild = []
        torightChild = []
        for issue in data:
            prediction = self.classifier.predict(issue[0])
            if prediction[0] is 0:
                issue[1].append(self.labelClasses[0])
                toleftChild.append(issue)
            else:
                issue[1].append(self.labelClasses[1])
                torightChild.append(issue)
        if self.rightChild is None or self.rightChild is None:
            return numpy.concatenate((toleftChild,torightChild,),axis= 0)
        return numpy.concatenate((self.leftChild.classify(toleftChild),self.rightChild.classify(torightChild)),axis= 0)
            

    



