import numpy
import logging

import load_data_antmap
import file_manipulation
import load_data
import label_classifier

categories:list = file_manipulation.FileManipulation.values["categories"] # [("doku", "bug")]#, ("doku", "api")]#, ("doku", "api"), ["doku", "bug", "enhancement"]]#, ("doku", "bug"), ("api", "bug")]
"""
    Description: This method is used to init the classifier using an antmap
"""

def initWithAntMap():
    amp: AntMapPreprozessor = load_data_antmap.AntMapPreprozessor()
    catIDX:int = 0

    for X_train, X_test, y_train, y_test in amp.getTrainingAndTestingData():
        cat:list[str] = categories[catIDX]
        lblClassif:LabelClassifier = label_classifier.LabelClassifier(cat)
        lblClassif.trainingClassifier(X_train, y_train)
        prediction:numpy.ndarray = lblClassif.predict(X_test)
        amp.prepareAntMap(prediction, y_test, X_train, [cat])
        logging.info("► ensemble-score:{}\n".format(numpy.mean(prediction == y_test)))

        catIDX += 1

"""
    Description: This method is used to init the classifier without using an Ant map
    Input:  loadClassifier :Bool  load the classifier
            saveClassifier :Bool  save the classifier
            loadVectorizer:Bool  load the vectorizer
    Output: 
"""
def initEverything():
    catIDX:int = 0
    processor = load_data.DataPreprocessor()
    for X_train, X_test, y_train, y_test in processor.getTrainingAndTestingData2():
        cat:list[str] = categories[catIDX]
        logging.info("\n--------- ( '{}', {} ) ---------".format(cat[0],str(cat[1:])))
        lblClassif:LabelClassifier = label_classifier.LabelClassifier(cat)
        lblClassif.trainingClassifier(X_train, y_train)
        prediction:numpy.ndarray = lblClassif.predict(X_test)
        lblClassif.accuracy(X_test, y_test, prediction)

        prediction2:numpy.ndarray = lblClassif.stackingPrediction(X_test)
        logging.info("► ensemble-score:{}\n".format(numpy.mean(prediction2 == y_test)))
        catIDX += 1



logging.basicConfig(level=logging.INFO)
logging.info('Started')
initEverything()
initWithAntMap()
logging.info('Finished')
