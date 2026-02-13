import os
from dotenv import load_dotenv
load_dotenv()

print(f"OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"GOOGLE_API_KEY present: {bool(os.getenv('GOOGLE_API_KEY'))}")

import config
print(f"Config loaded: {config.CONFIG['llm_settings']['provider']}")
