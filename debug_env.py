import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
if key:
    print(f"Key found. Length: {len(key)}")
    print(f"Prefix: {key[:7]}...")
    print(f"Suffix: ...{key[-4:]}")
else:
    print("KEY NOT FOUND")
