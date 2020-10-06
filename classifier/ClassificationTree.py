import loadClassifier
import numpy as np

class ClassificationTree:

    def __init__(self,lableClasses):
        self.rootNode = rootNode(lableClasses)
        
         
    
    def classify(self,data):
        classified =  self.rootNode.classify(data)
        sortedClassified = (sorted(classified, key = lambda element: element[2])) 
        return sortedClassified
        
     
    

class Node:

    def __init__(self,lableClasses,knowledge):
        self.lableClasses = lableClasses[0]
        self.classifier = loadClassifier.getClassifier(lableClasses.extend(knowledge))
        self.knowledge = knowledge

        if not lableClasses:
            self.leftChild = None
            self.rightChild = None
        else:
            self.leftChild = Node (lableClasses[1:],knowledge.append("{}".format(self.lableClasses)))
            self.rightChild =Node (lableClasses[1:],knowledge.append("not{}".format(self.lableClasses)))


    def classify (self,data):
        toleftChild = []
        torightChild = []
        for issue in data:
            prediction = self.classifier.predict(issue[0])
            if prediction[0] is 0:
                issue[1].append(self.lableClasses[0])
                toleftChild.append(issue)
            else:
                torightChild.append(issue)
        if self.rightChild is None or self.rightChild is None:
            return np.concatenate((toleftChild,torightChild,),axis= 0)
        return np.concatenate((self.leftChild.classify(toleftChild),self.rightChild.classify(torightChild)),axis= 0)
    
    def setRightChild(self, rightChild):
        self.rightChild = rightChild

    def setLeftChild(self, leftChild):
        self.leftChild = leftChild

class rootNode:
    def __init__(self, lableClasses):
        self.lableClasses = lableClasses[0:2]
        self.classifier = loadClassifier.getClassifier(lableClasses)
        self.leftChild = Node(lableClasses[2:],lableClasses[0])
        self.rightChild = Node(lableClasses[2:],lableClasses[1])
    
    def classify (self,data):
        toleftChild = []
        torightChild = []
        for issue in data:
            prediction = self.classifier.predict(issue[0])
            if prediction[0] is 0:
                issue[1].append(self.lableClasses[0])
                toleftChild.append(issue)
            else:
                issue[1].append(self.lableClasses[1])
                torightChild.append(issue)
        if self.rightChild is None or self.rightChild is None:
            return np.concatenate((toleftChild,torightChild,),axis= 0)
        return np.concatenate((self.leftChild.classify(toleftChild),self.rightChild.classify(torightChild)),axis= 0)
            

    



