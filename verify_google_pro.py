import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("No API Key found!")
else:
    genai.configure(api_key=api_key)
    try:
        print("Listing 1.5 models...")
        for m in genai.list_models():
            if '1.5' in m.name:
                print(f"Model: {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
