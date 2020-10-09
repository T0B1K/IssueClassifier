import json

import numpy as np


class FileManipulation:

    def __init__(self, output_folder="../auswertungen"):
        self.folder_name = "../documents"
        self.output_folder = output_folder

    def get_random_documents(self, label, elementcount):
        path = "{}/{}.json".format(self.folder_name, label)
        data = self.open_file(path, elementcount)
        permutation = np.random.permutation(data.shape[0])
        return data[permutation]

        # This method opens a file and returns all the documents
    def open_file(self, filename, elementcount=7000):
        with open(filename, "r") as file:
            data = file.read()
        # we just take all the "text" from the JSON
        documents = list(map(lambda entry: entry["text"], json.loads(data)))[
            :elementcount]
        return np.array(documents)

    def save_antmap_to_file(self, filename, data):
        path = self.output_folder+"/"+filename
        print(">\tsaving antmap in {}".format(path))
        f = open(path, "w", encoding='utf-8', errors='ignore')
        f.write(data)
        f.close()

    def save_misclassifications_to_file(self, filename, data):
        path = self.output_folder+"/"+filename
        print(">\tsaving Wrong Classified Texts in {}".format(path))
        f = open(path, "w", encoding='utf-8', errors='ignore')
        json_data = []
        for classified, document in data:
            json_data.append({
                "classified_as": classified,
                "text": document
            })
            # convert into JSON:
        f.write(json.dumps(json_data))
        f.close()
