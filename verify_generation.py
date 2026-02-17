import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=api_key)
    response = llm.invoke([HumanMessage(content="Hello")]) # No strip here initially
    print(f"Response: {response.content}")
except Exception as e:
    print(f"Error: {e}")
