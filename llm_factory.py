import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config import CONFIG, load_config

def get_llm(override_config=None):
    """
    Returns a configured LLM instance based on CONFIG or override_config.
    """
    conf = override_config if override_config else CONFIG
    
    settings = conf.get('llm_settings', {})
    provider = settings.get('provider', 'openai').lower()
    temperature = settings.get('temperature', 0.0)
    
    if provider == 'google':
        model_name = settings.get('google', {}).get('model', 'gemini-1.5-flash')
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key, max_retries=5)
    
    else: # Default to openai
        model_name = settings.get('openai', {}).get('model', 'gpt-4o-mini')
        return ChatOpenAI(model=model_name, temperature=temperature)

def get_embeddings(override_config=None):
    """
    Returns a configured Embeddings instance based on CONFIG or override_config.
    """
    conf = override_config if override_config else CONFIG
    
    settings = conf.get('llm_settings', {})
    provider = settings.get('provider', 'openai').lower()
    
    if provider == 'google':
        model_name = "models/gemini-embedding-001" # Corrected model from list_models
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    
    else: # Default to openai
        return OpenAIEmbeddings(model="text-embedding-3-small")

def reload_config_and_reinit():
    """
    Reloads the global CONFIG and clears any cached clients if we were caching them.
    For this simple script, re-reading CONFIG is enough if we re-instantiate classes.
    """
    global CONFIG
    CONFIG = load_config()
    return CONFIG
