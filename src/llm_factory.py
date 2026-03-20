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


# Maps provider name (as written in config.yaml) to its _try_* function.
_PROVIDER_FN_MAP = {
    "ollama":   _try_ollama,
    "groq":     _try_groq,
    "together": _try_together,
}


def _build_judge_chain(skip: list = None) -> list:
    """
    Returns an ordered list of (fn, name) tuples for the judge LLM resolution loop.

    Behaviour driven by config.yaml judge_llm section:
      - provider: "groq"    → use only Groq; error if unavailable
      - provider: "ollama"  → use only Ollama; error if unavailable
      - provider: "together"→ use only Together AI; error if unavailable
      - provider: "openai"  → use the primary OpenAI model directly
      - provider: "google"  → use the primary Google/Gemini model directly
      - provider: "auto"    → iterate auto_order list (default: ollama, groq, together)

    The optional `skip` list removes providers from the chain (used by the
    fallback function when a provider has been exhausted mid-run).
    """
    skip = {s.lower() for s in (skip or [])}
    judge_cfg = CONFIG.get('judge_llm', {})
    provider  = judge_cfg.get('provider', 'auto').lower().strip()

    if provider != 'auto':
        if provider in skip:
            # Requested provider was exhausted — fall through to auto_order
            pass
        elif provider in _PROVIDER_FN_MAP:
            return [(_PROVIDER_FN_MAP[provider], provider)]
        elif provider in ('openai', 'google'):
            return []  # signals caller to use secondary/primary path
        else:
            print(f"Warning: Unknown judge_llm.provider '{provider}' in config.yaml. Using auto.")

    # auto (or unknown/skipped explicit provider) — use auto_order
    auto_order = judge_cfg.get('auto_order', ['ollama', 'groq', 'together'])
    return [
        (_PROVIDER_FN_MAP[name], name)
        for name in auto_order
        if name in _PROVIDER_FN_MAP and name not in skip
    ]


def _resolve_secondary_or_primary(label_prefix: str = "") -> object:
    """Shared last-resort path: secondary provider → primary LLM."""
    global _judge_llm_label
    secondary = get_secondary_llm()
    if secondary:
        primary_provider = CONFIG.get('llm_settings', {}).get('provider', 'openai')
        sec_provider = 'google' if primary_provider == 'openai' else 'openai'
        sec_model = CONFIG.get('llm_settings', {}).get(sec_provider, {}).get('model', sec_provider)
        _judge_llm_label = f"{sec_provider}_{sec_model}"
        print(f"Judge LLM{label_prefix}: falling back to secondary provider ({sec_provider}).")
        return secondary

    print(f"Warning: No independent judge LLM available{label_prefix}. Using primary LLM.")
    primary_provider = CONFIG.get('llm_settings', {}).get('provider', 'openai')
    model = CONFIG.get('llm_settings', {}).get(primary_provider, {}).get('model', primary_provider)
    _judge_llm_label = f"{primary_provider}_{model}_primary"
    return get_llm()


def get_judge_llm():
    """
    Returns the LLM used for all independent judging tasks.

    Provider is controlled by config.yaml:
      judge_llm:
        provider: "auto"                        # or "ollama", "groq", "together", "openai", "google"
        auto_order: ["ollama", "groq", "together"]  # priority when provider is "auto"
    """
    judge_cfg = CONFIG.get('judge_llm', {})
    provider  = judge_cfg.get('provider', 'auto').lower().strip()

    # Explicit openai/google → skip straight to secondary/primary path
    if provider in ('openai', 'google'):
        return _resolve_secondary_or_primary()

    chain = _build_judge_chain()
    for fn, name in chain:
        llm = fn()
        if llm:
            return llm
        if provider != 'auto':
            # Explicit provider failed — surface a clear error instead of silently falling back
            print(f"Error: Requested judge provider '{provider}' is unavailable. "
                  f"Check your setup or set judge_llm.provider to 'auto' in config.yaml.")
            break

    return _resolve_secondary_or_primary()


def get_fallback_judge_llm(skip: list = None):
    """
    Returns a judge LLM skipping any exhausted providers.
    Called automatically mid-run when a provider hits its quota or rate limit.

    `skip` is a list of provider names to exclude, e.g. ['groq'].
    When not provided, falls back to auto_order minus Groq (historical default).
    """
    if skip is None:
        skip = ['groq']
    chain = _build_judge_chain(skip=skip)
    for fn, _ in chain:
        llm = fn()
        if llm:
            return llm

    return _resolve_secondary_or_primary(label_prefix=" fallback")


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
