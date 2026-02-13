import pandas as pd
from rcm_engine import RcmAuditor
from config import CONFIG
import json

def test_single_row():
    print("Initializing Auditor...")
    auditor = RcmAuditor()
    print("Building RAG Index...")
    auditor.initialize_rag()

    # Load CSV
    input_csv = CONFIG['paths']['input_csv']
    df = pd.read_csv(input_csv, sep=';')
    
    # Get Row 3 (Control Ref 1.3)
    target_row = df[df['#'] == 3.0]
    
    if target_row.empty:
        print("Row 3 not found in CSV!")
        # Fallback to index 15 as seen in line 16 of csv view?
        # NO, '28' is strictly in column '#'. 
        # Looking at csv content from view_file:
        # line 16: 28;1;PD;6.3;Is lifetime PD estimation well...
        # So it is there.
        return

    row_dict = target_row.iloc[0].to_dict()
    print(f"Testing Control Ref: {row_dict.get('Control Reference')}")
    print(f"Question: {row_dict.get('Design Effectiveness Assessment')}")

    result = auditor.process_row(row_dict)
    
    print("\n=== AI Answer ===")
    print(result['AI_Answer'])
    print("\n=== Validation Score ===")
    print(result['Validation_Score'])
    
    if "Not Documented" not in result['AI_Answer']:
        print("\nSUCCESS: Answer found!")
    else:
        print("\nFAILURE: Still returning 'Not Documented'.")

if __name__ == "__main__":
    test_single_row()
