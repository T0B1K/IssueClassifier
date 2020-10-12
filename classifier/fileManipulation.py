import json
import joblib
import numpy as np



""" Class provies methods for loading and writthing files with additional parameters """


class FileManipulation:
    
    values = json.loads(open("TrainingConfig.txt").read())
    
    """Description: Constructor for FileManipulation
        Input:  outputFolder optional path parameter 
        Output: Return FileManipulation object"""

    def __init__(self, outputFolder="."):
        self.outputFolder = outputFolder
        self.folderName = "../issues"

    def getRandomDocs(self, label, elementcount):
        """Description: get random Documents a class
        Input:  label of the document class
                elementcount amount of documents to load 
        Output: Return List[String] of documents"""
        path = "{}/{}.json".format(self.folderName, label)
        data = self.openFile(path, elementcount)
        perm = np.random.permutation(data.shape[0])
        return data[perm]

        # This method opens a file and returns all the documents
    def openFile(self, filename, elementcount=values["elementcount"]):
        """
        Description: Method loads file 
        Input:  filename name of the file
                elementcount amount of documents to load optional
        Output: Return List[String] of documents
        """
        with open(filename, "r") as file:
            data = file.read()
        # we just take all the "text" from the JSON
        documents = list(map(lambda entry: entry["text"], json.loads(data)))[
            :elementcount]
        return np.array(documents)

    def saveAntmapToFile(self, filename, data):
        """
        Description: Method to save Antmap to file
        Input:  filename name of the file
                data to save
        Output: Return nothing
        """
        path = self.outputFolder+"/"+filename
        print(">\tsaving antmap in {}".format(path))
        f = open(path, "w", encoding='utf-8', errors='ignore')
        f.write(data)
        f.close()

    def saveWrongClassifiedToFile(self, filename, data):
        """
        Description: Method to save wrong Classified 
        Input:  filename name of the file
                data to save
        Output: Nothing
        """
        path = filename  # self.outputFolder+"/"+filename
        print(">\tsaving Wrong Classified Texts in {}".format(path))
        f = open(path, "w", encoding='utf-8', errors='ignore')
        jsonData = []
        for classified, document in data:
            jsonData.append({
                "classified_as": classified,
                "text": document
            })
            # convert into JSON:
        f.write(json.dumps(jsonData))
        f.close()
