import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

# Project root: parent of src/ (where this file resides)
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config():
    load_dotenv(PROJECT_ROOT / ".env")

    if not CONFIG_PATH.exists():
        # Fallback if config.py is somehow in root (during transition or wrong usage)
        if os.path.exists("config.yaml"):
             return load_config_from_path("config.yaml")
             
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    # Make paths absolute based on PROJECT_ROOT
    if 'paths' in config:
        for key, path in config['paths'].items():
            # Only prepend if not already absolute
            if not os.path.isabs(path):
                config['paths'][key] = str(PROJECT_ROOT / path)
            
    return config

# Helper for fallback
def load_config_from_path(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

try:
    CONFIG = load_config()
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    CONFIG = {}
