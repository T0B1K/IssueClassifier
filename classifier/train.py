import numpy
import logging

import load_data_antmap
import file_manipulation
import load_data
import label_classifier

labelClasses = ["enhancement", "bug", "doku", "api"]
categories = [("doku", "bug")]#, ("doku", "api")]#, ("doku", "api"), ["doku", "bug", "enhancement"]]#, ("doku", "bug"), ("api", "bug")]
"""
    Description: This method is used to init the classifier using an antmap
    Input:  loadClassifier :Bool load the classifier
            saveClassifier :Bool save the classifier
    Output: 
"""

def initWithAntMap(loadClassifier = False, saveClassifier = False):
    amp = load_data_antmap.AntMapPreprozessor(labelClasses, categories)
    catIDX = 0

    for X_train, X_test, y_train, y_test in amp.getTrainingAndTestingData(labelClasses, categories):
        cat = categories[catIDX]
        lblClassif = label_classifier.LabelClassifier(cat)
        lblClassif.trainingClassifier(X_train, y_train, loadClassifier, saveClassifier)
        prediction = lblClassif.predict(X_test)
        amp.createAntMapAndDocumentView(prediction, y_test, X_train, [cat])
        logging.info("► ensemble-score:{}\n".format(numpy.mean(prediction == y_test)))

        catIDX += 1

"""
    Description: This method is used to init the classifier without using an Ant map
    Input:  loadClassifier :Bool  load the classifier
            saveClassifier :Bool  save the classifier
            loadVectorizer:Bool  load the vectorizer
    Output: 
"""
def initEverything(loadClassifier = False, saveClassifier = False, loadVectorizer = True):
    catIDX = 0
    processor = load_data.DataPreprocessor(labelClasses, categories, loadVectorizer)
    for X_train, X_test, y_train, y_test in processor.getTrainingAndTestingData2():#labelClasses, categories):
        cat = categories[catIDX]
        logging.info("\n--------- ( '{}', {} ) ---------".format(cat[0],str(cat[1:])))
        lblClassif = label_classifier.LabelClassifier(cat)
        lblClassif.trainingClassifier(X_train, y_train, loadClassifier)
        prediction = lblClassif.predict(X_test)
        lblClassif.accuracy(X_test, y_test, prediction)

        prediction2 = lblClassif.stackingPrediction(X_test)
        logging.info("► ensemble-score:{}\n".format(numpy.mean(prediction2 == y_test)))
        #hue.createAntMapAndDocumentView(prediction, y_test, X_train, [cat])
        catIDX += 1



logging.basicConfig(level=logging.INFO)
logging.info('Started')
#initEverything()
initWithAntMap()
logging.info('Finished')
