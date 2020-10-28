import json

class Configuration():
    
    configValues = json.loads(open("load_config.json").read())

    def __init__(self):
        """Description: Constructor for Configuration""" 

    def getValueFromConfig(self, configPath):
        """Description: Splits the input string into an array of key values and iterates through the configuration JSON.
        Input: String containing the keys of the desired configuration value.
        Output: Returns the desired configuration value as a string."""
        path = configPath.split()
        config = self.configValues
        for key in path:
            config = config[key]
        return config