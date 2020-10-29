import json

class Configuration():
    
    configValues = json.loads(open("load_config.json").read())

    def __init__(self):
        """Constructor for Configuration
        """ 

    def getValueFromConfig(self, configPath: str):
        """Splits the input string into an array of key values and iterates through the configuration JSON.

        Args:
            configPath (str): String containing the keys of the desired configuration value.

        Returns:
            [type]: Returns the desired configuration value as a string or array.
        """
        path = configPath.split()
        config = self.configValues
        for key in path:
            config = config[key]
        return config