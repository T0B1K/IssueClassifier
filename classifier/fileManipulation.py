import json
import joblib
import numpy as np

class FileManipulation:
    def __init__(self, outputFolder="../auswertungen"):
        self.folderName = "../documents"
        self.outputFolder = outputFolder

    def getRandomDocs(self, label, elementcount):
        path = "{}/{}.json".format(self.folderName, label)
        data = self.openFile(path, elementcount)
        perm = np.random.permutation(data.shape[0]) 
        return data[perm]
    
        # This method opens a file and returns all the documents
    def openFile(self, filename, elementcount=4000):
        with open(filename, "r") as file:
            data = file.read()
        # we just take all the "text" from the JSON
        documents = np.array(list(map(lambda entry: entry["text"], json.loads(data))))
        return documents[:elementcount]