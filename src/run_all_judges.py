import os
import sys
import time
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config
from validate_audit import validate_audit

def main():
    # The explicit output folder you wanted to use
    target_folder = r"C:\Users\semeier\Desktop\gemini_chat_private_GH\Agentic_AI_for_validation\outputs\20260324T144520Z"
    
    if not os.path.exists(target_folder):
        print(f"Error: Target folder does not exist: {target_folder}")
        return

    # 1. Groq, 2. Together AI, 3. Ollama local
    providers = ["groq"] #["groq", "together", "ollama"]

    for provider in providers:
        print(f"\n" + "="*60)
        print(f"  RUNNING VALIDATION WITH JUDGE LLM: {provider.upper()}")
        print("="*60)
        
        # Override the provider in the globally loaded config memory
        config.CONFIG['judge_llm']['provider'] = provider
        
        try:
            start_time = time.time()
            # Pass the target_folder directly to validate_audit to skip auto-detect
            validate_audit(run_folder=target_folder)
            elapsed = time.time() - start_time
            print("\n--------------------------------------------------")
            print(f"✓ FINISHED {provider.upper()} in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"Error occurred while running with {provider}: {e}")

if __name__ == "__main__":
    main()
