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
        with open("models.txt", "w") as f:
            for m in genai.list_models():
                f.write(f"{m.name}\n")
        print("Models listed to models.txt")
    except Exception as e:
        print(f"Error listing models: {e}")
