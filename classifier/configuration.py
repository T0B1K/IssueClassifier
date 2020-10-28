import json

class Configuration():
    
    configValues = json.loads(open("load_config.json").read())

    def __init__(self):
        """Description: Constructor for Configuration""" 

    def resolvePath(self, config, path):
        """Description: Iterates recursively through the configuration tree to a value.
        Input: Array of JSON keys and the configuration JSON.
        Output: String containing the desired configuration value."""
        if len(path) == 0:
            return config
        else:
            config = config[path[0]]
            path.pop(0) 
            return self.resolvePath(config, path)

    def getValueFromConfig(self, configPath):
        """Description: Splits the input string into an array and passes it to the resolvePath() function.
        Input: String containing the keys of the desired configuration value.
        Output: Returns the desired configuration value as a string."""
        path = configPath.split()
        return self.resolvePath(self.configValues, path)