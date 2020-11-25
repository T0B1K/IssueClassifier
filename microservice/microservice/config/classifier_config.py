import json
from typing import Any


class Configuration:
    _config_values = json.loads(
        open("/microservice/microservice/config/load_config.json").read()
    )

    def get_value_from_config(self, config_path: str) -> Any:
        if config_path == "":
            raise Exception("No config path was given")
        try:
            path = config_path.split()
            config = self._config_values
            for key in path:
                config = config[key]
            return config
        except Exception:
            raise KeyError(
                "Invalid configuration key! Compare your Input to the load_config.json keys!"
            )
