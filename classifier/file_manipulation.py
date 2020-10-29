import json
import joblib
import numpy
import logging

import configuration

config = configuration.Configuration()


class FileManipulation:
    """ Class provies methods for loading and writing files with additional parameters """

    def __init__(self):
        """Constructor for FileManipulation
        """

    def getRandomDocs(self, label: str, elementcount: int) -> numpy.ndarray:
        """This method is used for getting random documents from a labelclass.

        Args:
            label (str): The label i.e. "bug"
            elementcount (int): How many documents should be randomly chosen from the corresponding file.

        Returns:
            numpy.ndarray: The randomly chosen documents of the specific file.
        """
        if label == "":
            raise("The label is not being used")
        if elementcount < 1:
            raise("The elementcount provided was less or equal to zero")
        path: str = "{}/{}.json".format(
            config.getValueFromConfig("issueFolder"), label)
        data: numpy.ndarray = self.openFile(path, elementcount)
        assert data.size, "No data was loaded"
        perm: numpy.ndarray = numpy.random.permutation(data.shape[0])
        assert perm.size, "No permutation was given"
        permutedArray: numpy.ndarray = data[perm]
        assert permutedArray.size, "An error occurrsed during shuffleing "
        return permutedArray

    def openFile(self, filename: str, elementcount: int = config.getValueFromConfig("trainingConstants elementCount")) -> numpy.ndarray:
        """Method loads a file from the path provided in the config.

        Args:
            filename (str): The name of the file.
            elementcount (int, optional): The amount of elements of the loaded documents. Defaults to config.getValueFromConfig("trainingConstants elementCount").

        Returns:
            numpy.ndarray: Array of loaded documents
        """
        if elementcount < 1:
            raise("Not enough elements - elementcount was less or equal to zero")
        if filename == "":
            raise("No filename provided")
        with open(filename, "r") as file:
            data = file.read()
        # we just take all the "text" from the JSON
        documents: list = list(map(lambda entry: entry["text"], json.loads(data)))[
            :elementcount]
        documentArray: numpy.ndarray = numpy.array(documents)
        assert documentArray.size, "No documents were provided"
        return documentArray

    def saveAntmapToFile(self, filename: str, data: str):
        """This method is used to save the antmap to a file.

        Args:
            filename (str): The name of the file, where to store the antmap.
            data (str): The data of the antmap (the antmap itself).
        """
        if data == "":
            raise("No data has been provided")
        if filename == "":
            raise("No filename has been provided")
        path: str = "{}/{}".format(
            config.getValueFromConfig("outputFolder"), filename)
        logging.info(">\tsaving antmap in {}".format(path))
        with open(path, "w", encoding='utf-8', errors='ignore') as f:
            f.write(data)

    def saveWrongClassifiedToFile(self, filename: str, data: str):
        """This method saves the wrong classified issues to a file

        Args:
            filename (str): The filename of the file
            data (str): The wrong classified issues
        """
        if data == "":
            raise("No data has been provided")
        if filename == "":
            raise("No filename has been provided")
        path: str = filename  # self.outputFolder+"/"+filename
        logging.info(">\tsaving Wrong Classified Texts in {}".format(path))
        with open(path, "w", encoding='utf-8', errors='ignore') as f:
            jsonData: list = []
            for classified, document in data:
                jsonData.append({
                    "classified_as": classified,
                    "text": document
                })
                # convert into JSON:
            f.write(json.dumps(jsonData))
            f.close()
