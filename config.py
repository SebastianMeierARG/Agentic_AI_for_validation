import os
import yaml
from dotenv import load_dotenv

def load_config(config_path="config.yaml"):
    load_dotenv()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

# Global config object
try:
    CONFIG = load_config()
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    CONFIG = {}
