import json
import joblib
import numpy
import logging

import configuration

config = configuration.Configuration()


""" Class provies methods for loading and writthing files with additional parameters """


class FileManipulation:
    
    def __init__(self):
        """Description: Constructor for FileManipulation
        Input:  outputFolder optional path parameter 
        Output: Return FileManipulation object"""

    def getRandomDocs(self, label:str, elementcount:int) -> numpy.ndarray:
        """Description: get random Documents a class
        Input:  label of the document class
                elementcount amount of documents to load 
        Output: Return List[String] of documents"""
        path:str = "{}/{}.json".format(config.getValueFromConfig("issueFolder"), label)
        data:numpy.ndarray = self.openFile(path, elementcount)
        perm:numpy.ndarray = numpy.random.permutation(data.shape[0])
        return data[perm]

    def openFile(self, filename:str, elementcount:int=config.getValueFromConfig("trainingConstants elementCount")) -> numpy.ndarray:
        """
        Description: Method loads file 
        Input:  filename name of the file
                elementcount amount of documents to load optional
        Output: Return List[String] of documents
        """
        with open(filename, "r") as file:
            data = file.read()
        # we just take all the "text" from the JSON
        documents:list = list(map(lambda entry: entry["text"], json.loads(data)))[
            :elementcount]
        return numpy.array(documents)

    def saveAntmapToFile(self, filename:str, data:str):
        """
        Description: Method to save Antmap to file
        Input:  filename name of the file
                data to save
        Output: Return nothing
        """
        path:str = "{}/{}".format(config.getValueFromConfig("outputFolder"), filename)
        logging.info(">\tsaving antmap in {}".format(path))
        with open(path, "w", encoding='utf-8', errors='ignore') as f:
            f.write(data)

    def saveWrongClassifiedToFile(self, filename:str, data:str):
        """
        Description: Method to save wrong Classified 
        Input:  filename name of the file
                data to save
        Output: Nothing
        """
        path:str = filename  # self.outputFolder+"/"+filename
        logging.info(">\tsaving Wrong Classified Texts in {}".format(path))
        with open(path, "w", encoding='utf-8', errors='ignore') as f:
            jsonData:list = []
            for classified, document in data:
                jsonData.append({
                    "classified_as": classified,
                    "text": document
                })
                # convert into JSON:
            f.write(json.dumps(jsonData))
            f.close()
