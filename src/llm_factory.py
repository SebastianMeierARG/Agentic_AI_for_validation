import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config import CONFIG, load_config

# Stores a human-readable label for whichever judge LLM was actually initialised.
# Readable via get_judge_llm_label() after calling get_judge_llm().
_judge_llm_label: str = "unknown"


def get_judge_llm_label() -> str:
    """Return a short, filename-safe label for the judge LLM that was last initialised."""
    return _judge_llm_label

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
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            api_key = api_key.strip()
        return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)

def get_embeddings(override_config=None):
    """
    Returns a configured Embeddings instance based on CONFIG or override_config.
    """
    conf = override_config if override_config else CONFIG
    
    settings = conf.get('llm_settings', {})
    provider = settings.get('provider', 'openai').lower()
    
    if provider == 'google':
        model_name = settings.get('google', {}).get('embedding_model', 'models/embedding-001')
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    
    else: # Default to openai
        model_name = settings.get('openai', {}).get('embedding_model', 'text-embedding-3-small')
        api_key = os.getenv("OPENAI_API_KEY")
        kwargs = {"model": model_name}
        if api_key:
            kwargs["openai_api_key"] = api_key.strip()
        return OpenAIEmbeddings(**kwargs)

def _try_groq() -> object:
    """Attempt to initialise a Groq LLM. Returns None on any failure."""
    global _judge_llm_label
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None
    try:
        from langchain_groq import ChatGroq
        model_name = CONFIG.get('judge_llm', {}).get('model', 'llama-3.3-70b-versatile')
        llm = ChatGroq(model=model_name, temperature=0.0, groq_api_key=groq_api_key)
        _judge_llm_label = f"groq_{model_name.replace('/', '-')}"
        print(f"Judge LLM: Groq / {model_name}")
        return llm
    except ImportError:
        print("Warning: langchain-groq not installed. Run: pip install langchain-groq")
    except Exception as e:
        print(f"Warning: Could not initialise Groq: {e}")
    return None


def _try_together() -> object:
    """Attempt to initialise Together AI (free tier). Returns None on any failure."""
    global _judge_llm_label
    api_key = os.getenv("TOGETHER_API_KEY", "").strip()
    if not api_key:
        return None
    model_name = CONFIG.get('judge_llm', {}).get(
        'together_model', 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'
    )
    # Prefer the dedicated langchain-together package which handles auth correctly.
    # Falls back to ChatOpenAI with explicit headers if the package is not installed.
    try:
        from langchain_together import ChatTogether
        llm = ChatTogether(model=model_name, together_api_key=api_key, temperature=0.0)
        _judge_llm_label = f"together_{model_name.split('/')[-1].replace(' ', '-')}"
        print(f"Judge LLM: Together AI / {model_name}")
        return llm
    except ImportError:
        pass  # fall through to ChatOpenAI approach
    except Exception as e:
        print(f"Warning: Could not initialise Together AI via ChatTogether: {e}")
        return None
    # Fallback: ChatOpenAI with explicit Authorization header to avoid key-mixing
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )
        _judge_llm_label = f"together_{model_name.split('/')[-1].replace(' ', '-')}"
        print(f"Judge LLM: Together AI / {model_name} (via ChatOpenAI)")
        return llm
    except Exception as e:
        print(f"Warning: Could not initialise Together AI: {e}")
    return None


def _try_ollama() -> object:
    """
    Attempt to initialise a local Ollama LLM. Returns None if Ollama is not running.
    Install: https://ollama.com  |  Pull model: ollama pull llama3.2
    """
    global _judge_llm_label
    try:
        from langchain_ollama import ChatOllama
        judge_cfg  = CONFIG.get('judge_llm', {})
        model_name = judge_cfg.get('ollama_model', 'llama3.2')
        base_url   = judge_cfg.get('ollama_base_url', 'http://localhost:11434')
        llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.0)
        llm.invoke("ping")  # probe — fails silently if Ollama not running
        _judge_llm_label = f"ollama_{model_name.replace(':', '-')}"
        print(f"Judge LLM: Ollama (local) / {model_name}")
        return llm
    except ImportError:
        print("Warning: langchain-ollama not installed. Run: pip install langchain-ollama")
    except Exception:
        pass
    return None


def get_judge_llm():
    """
    Returns the LLM used for all independent judging tasks.

    Fallback chain (first available wins):
      1. Groq         — free cloud, fast; needs GROQ_API_KEY; 100k tokens/day free tier
      2. Together AI  — free cloud model; needs TOGETHER_API_KEY (console.together.ai)
      3. Ollama       — local, zero cost/rate-limits; needs Ollama running locally
      4. Secondary provider (Google/OpenAI, whichever is not the primary)
      5. Primary LLM  — last resort; least independent
    """
    for fn, label in [(_try_groq, "Groq"), (_try_together, "Together AI"), (_try_ollama, "Ollama")]:
        llm = fn()
        if llm:
            return llm

    secondary = get_secondary_llm()
    if secondary:
        global _judge_llm_label
        provider = CONFIG.get('llm_settings', {}).get('provider', 'openai')
        secondary_provider = 'google' if provider == 'openai' else 'openai'
        sec_model = CONFIG.get('llm_settings', {}).get(secondary_provider, {}).get('model', secondary_provider)
        _judge_llm_label = f"{secondary_provider}_{sec_model}"
        print("Judge LLM: falling back to secondary provider (Google/OpenAI).")
        return secondary

    print("Warning: No independent judge LLM available. Using primary LLM for judging.")
    provider = CONFIG.get('llm_settings', {}).get('provider', 'openai')
    model = CONFIG.get('llm_settings', {}).get(provider, {}).get('model', provider)
    _judge_llm_label = f"{provider}_{model}_primary"
    return get_llm()


def get_fallback_judge_llm():
    """
    Returns a judge LLM that explicitly skips Groq (used when Groq TPD quota is exhausted).
    Fallback chain: Together AI → Ollama → secondary provider → primary LLM.
    """
    for fn in [_try_together, _try_ollama]:
        llm = fn()
        if llm:
            return llm

    secondary = get_secondary_llm()
    if secondary:
        global _judge_llm_label
        provider = CONFIG.get('llm_settings', {}).get('provider', 'openai')
        secondary_provider = 'google' if provider == 'openai' else 'openai'
        sec_model = CONFIG.get('llm_settings', {}).get(secondary_provider, {}).get('model', secondary_provider)
        _judge_llm_label = f"{secondary_provider}_{sec_model}"
        print("Judge LLM fallback: switching to secondary provider (Google/OpenAI).")
        return secondary

    print("Warning: No secondary provider available. Using primary LLM as judge fallback.")
    provider = CONFIG.get('llm_settings', {}).get('provider', 'openai')
    model = CONFIG.get('llm_settings', {}).get(provider, {}).get('model', provider)
    _judge_llm_label = f"{provider}_{model}_primary"
    return get_llm()


def get_secondary_llm():
    """
    Returns an LLM from the *other* provider for cross-validation.
    If the primary is OpenAI, returns a Google LLM (and vice versa).
    Returns None if the secondary provider's API key is not available.
    """
    provider = CONFIG.get('llm_settings', {}).get('provider', 'openai').lower()

    if provider == 'openai':
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        model_name = CONFIG.get('llm_settings', {}).get('google', {}).get('model', 'gemini-1.5-flash')
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key, max_retries=3)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        model_name = CONFIG.get('llm_settings', {}).get('openai', {}).get('model', 'gpt-4o-mini')
        return ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=api_key.strip())


def reload_config_and_reinit():
    """
    Reloads the global CONFIG and clears any cached clients if we were caching them.
    For this simple script, re-reading CONFIG is enough if we re-instantiate classes.
    """
    global CONFIG
    CONFIG = load_config()
    return CONFIG
