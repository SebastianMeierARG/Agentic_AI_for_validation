# this file runs the pipeline for the project
import os   
from generate_performance_report import generate_report
from validate_audit import validate_audit
import argparse
# now i run the ./src/run_audit.py and then validate the results

def run_pipeline():
    print("Running the audit...", flush=True)
    os.system('python src/run_audit.py')

    print("Audit completed. Now validating the results...", flush=True)
    validate_audit()
    
    print("Creating the Validation Report.", flush=True)
    # os.system('python src/generate_performance_report.py')

if __name__ == "__main__":
    run_pipeline()