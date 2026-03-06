import os
import shutil
import time
from dotenv import load_dotenv
from rcm_engine import RcmAuditor
from config import CONFIG
import pandas as pd
import json

# Load env vars
load_dotenv()

# Provider is determined by config.yaml
print(f"Current provider: {CONFIG['llm_settings']['provider']}")

def test_optimization_verification():
    print("=== Phase 1: Persistent Vector Store Verification ===")
    
    # Ensure no index exists first
    if os.path.exists("faiss_index_client"):
        print("Removing existing index for clean test...")
        shutil.rmtree("faiss_index_client")
    
    # Re-init factory with new config provider (if necessary, though CONFIG is shared)
    from llm_factory import get_llm, get_embeddings
    
    # We might need to reload modules if factory caches config, but usually it reads CONFIG directly.
    # rag_engine reads CONFIG.
    
    auditor = RcmAuditor() 
    
    start_time = time.time()
    print("Building index from scratch (1st run)...")
    try:
        auditor.initialize_rag()
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        return

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    assert os.path.exists("faiss_index_client"), "Index folder was not created!"
    
    # 2nd run - should load from disk
    print("\nRe-initializing Auditor to test loading from disk (2nd run)...")
    auditor_2 = RcmAuditor()
    start_time = time.time()
    auditor_2.initialize_rag()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    print("\n=== Phase 2: Client Summary verification ===")
    if os.path.exists("client_summary.txt"):
        os.remove("client_summary.txt")
        
    auditor_2.generate_client_summary()
    if os.path.exists("client_summary.txt"):
         with open("client_summary.txt", "r", encoding="utf-8") as f:
            summary = f.read()
            print(f"Summary length: {len(summary)} chars")
            if len(summary) > 100:
                print("Client Summary Verified.")
            else:
                print("Warning: Summary is too short!")
    else:
        print("Error: Client summary file not created!")

    print("\n=== Phase 3: Compliance Verdict Verification ===")
    # Create a mock row
    mock_row = {
        'Control Reference': 'TEST-REF-001',
        'Design Effectiveness Assessment': 'Does the bank have a documented policy for IFRS 9?',
        'Test Procedures': 'Verify the existence of the policy.',
        'Test Procedure': 'Verify the existence of the policy.'
    }
    
    print("Processing mock row...")
    try:
        result = auditor_2.process_row(mock_row)
        print("\nResult Keys:", result.keys())
        if 'Compliance_Verdict' in result:
             print(f"Verdict: {result['Compliance_Verdict']}")
             print("Compliance Verdict Verified.")
        else:
             print("Error: Compliance_Verdict column missing!")
        
        print(f"AI Answer: {result.get('AI_Answer', 'N/A')[:100]}...")
    except Exception as e:
        print(f"Error processing row: {e}")
    
    print("\n=== ALL TESTS COMPLETED ===")

if __name__ == "__main__":
    test_optimization_verification()
