import yaml 

class Config: 
    _instance = None

    def __new__(cls, config_path: str):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def __getattr__(self, name: str):
        if name not in self.config.keys():
            raise AttributeError(f"Attribute {name} not defined")
        return self.config[name]

