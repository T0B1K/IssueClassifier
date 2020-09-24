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
    def openFile(self, filename, elementcount=1000):
        with open(filename, "r") as file:
            data = file.read()
        # we just take all the "text" from the JSON
        documents = list(map(lambda entry: entry["text"], json.loads(data)))[:elementcount]
        return np.array(documents)
    
    def saveAntmapToFile(self, filename, data):
        path = self.outputFolder+"/"+filename
        print(">\tsaving antmap in {}".format(path))
        f = open(path, "w",encoding='utf-8', errors='ignore')
        f.write(data)
        f.close()
    
    def saveWrongClassifiedToFile(self, filename, data):
        path = self.outputFolder+"/"+filename
        print(">\tsaving Wrong Classified Texts in {}".format(path))
        f = open(path, "w",encoding='utf-8', errors='ignore')
        jsonData = []
        for classified, document in data:
            jsonData.append({
                "classified_as": classified,
                "text": document
            })
            # convert into JSON:
        f.write(json.dumps(jsonData))
        f.close()