import os
from config import CONFIG
from rcm_engine import RcmAuditor
import pandas as pd
import json

def main():
    print("Starting Audit Process...")
    
    # Initialize Auditor
    auditor = RcmAuditor()
    print("Initializing RAG Engine (this may take a moment)...")
    auditor.initialize_rag()
    
    # Load Input
    input_csv = CONFIG['paths']['input_csv']
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found.")
        return

    print(f"Reading input from {input_csv}...")
    try:
        # Try utf-8 first
        df = pd.read_csv(input_csv, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Attempting with fallback encoding (windows-1252)...")
        try:
            df = pd.read_csv(input_csv, sep=';', encoding='windows-1252')
        except Exception as e:
            print(f"Error reading CSV with fallback encoding: {e}")
            return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    results = []
    total_rows = len(df)
    print(f"Processing {total_rows} rows...")
    
    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{total_rows}...")
        try:
            row_dict = row.to_dict()
            res = auditor.process_row(row_dict)
            results.append(res)
        except Exception as e:
            print(f"Error processing row {idx + 1}: {e}")
            # Add error info to result
            err_row = row.to_dict()
            err_row['AI_Answer'] = f"Error: {e}"
            results.append(err_row)
            
        # Polite delay between rows to avoid hitting rate limits
        import time
        time.sleep(1)

    # Save Results
    output_json = CONFIG['paths']['output_json']
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Audit complete. Results saved to {output_json}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
