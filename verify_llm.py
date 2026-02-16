import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded Key Repr: {repr(api_key)}")

if api_key:
    api_key_stripped = api_key.strip()
    print(f"Stripped Key Repr: {repr(api_key_stripped)}")
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key_stripped)
        response = llm.invoke([HumanMessage(content="Hello")])
        print("Success!")
        print(response.content)
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No key found.")
