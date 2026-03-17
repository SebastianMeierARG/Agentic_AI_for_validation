import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from rcm_engine import RcmAuditor
from config import PROJECT_ROOT

if __name__ == "__main__":
    print("Starting Client Summary Generation...", flush=True)
    auditor = RcmAuditor()
    print("Initializing RAG Engine...", flush=True)
    auditor.initialize_rag()
    # Force regeneration by removing existing file
    summary_path = os.path.join(PROJECT_ROOT, "outputs", "client_summary.md")
    if os.path.exists(summary_path):
        os.remove(summary_path)
        print("Removed existing summary — regenerating.", flush=True)
    auditor.generate_client_summary()
    print("Client Summary generation complete.", flush=True)
