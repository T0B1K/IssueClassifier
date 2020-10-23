import numpy as np


import loadDataAntmap
import fileManipulation
import loadData
import label_classifier

labelClasses = ["enhancement", "bug", "doku", "api"]
categories = [("doku", "bug")]
trainingPercentage = fileManipulation.FileManipulation.values["trainingPercentage"]  

"""
    Description: This method is used to init the classifier using an antmap
    Input:  loadClassifier :Bool load the classifier
            saveClassifier :Bool save the classifier
    Output: 
"""
def initWithAntMap(loadClassifier = False, saveClassifier = False):
    amp = loadDataAntmap.AntMapPreprozessor(labelClasses, categories)
    catIDX = 0

    for X_train, X_test, y_train, y_test in amp.getTrainingAndTestingData(labelClasses, categories):
        cat = categories[catIDX]
        lblClassif = label_classifier.LabelClassifier(cat)
        lblClassif.trainClassifier(X_train, y_train, loadClassifier, saveClassifier)
        prediction = lblClassif.predict(X_test)
        amp.createAntMapAndDocumentView(prediction, y_test, X_train, [cat])
        print("->> ensemble-score:{}\n".format(np.mean(prediction == y_test)))

        catIDX += 1

"""
    Description: This method is used to init the classifier without using an Ant map
    Input:  loadClassifier :Bool  load the classifier
            saveClassifier :Bool  save the classifier
            loadVectorrizer:Bool  load the vectorrizer
    Output: 
"""
def initEverything(loadClassifier = False, saveClassifier = False, loadVectorrizer = True):
    catIDX = 0
    processor = loadData.DataPreprocessor(labelClasses, categories, loadVectorrizer)
    for X_train, X_test, y_train, y_test in processor.getTrainingAndTestingData2():
        cat = categories[catIDX]
        print("\n--------- ( '{}', {} ) ---------".format(cat[0],str(cat[1:])))
        lblClassif = label_classifier.LabelClassifier(cat)
        lblClassif.trainClassifier(X_train, y_train, loadClassifier)
        prediction = lblClassif.predict(X_test)
        lblClassif.accuracy(X_test, y_test, prediction)

        prediction2 = lblClassif.stackingPrediction(X_test)
        print("► ensemble-score:{}\n".format(np.mean(prediction2 == y_test)))
        catIDX += 1


#initEverything()
initWithAntMap()
