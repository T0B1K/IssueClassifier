import json

class Configuration():
    
    configValues = json.loads(open("load_config.json").read())

    def __init__(self):
        """Constructor for Configuration
        """ 

    def getValueFromConfig(self, configPath: str) -> any:
        """Splits the input string into an array of key values and iterates through the configuration JSON.

        Args:
            configPath (str): String containing the keys of the desired configuration value.

        Returns:
            [type]: Returns the desired configuration value.
        """
        try:
            path = configPath.split()
            config = self.configValues
            for key in path:
                config = config[key]
            return config
        except:
            raise KeyError("Invalid configuration key! Compare your Input to the load_config.json keys!")