# IFRS9 Automated Auditor

An AI tool for Credit Risk Validation that ingests an RCM (Risk Control Matrix) CSV, retrieves evidence from PDFs using RAG, answers audit questions, and validates its own answers.

## Setup

1.  **Dependencies**:
    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    *   Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Edit `.env` and add your API keys (e.g., `OPENAI_API_KEY`).
    *   Review `config.yaml` to adjust settings if needed (model, paths, validation).

3.  **Documents**:
    *   Place your PDF documents in the `documents/` folder. The system will automatically ingest and index them.

4.  **Input Data**:
    *   Place your RCM CSV file at the path specified in `config.yaml` (default: `rcm_input.csv`).
    *   The CSV should have at least `Control Reference` and `Test Procedure` columns.

## Interactive Testing

To test the system without running a full audit:

1.  Open the `interactive_audit.ipynb` notebook in Jupyter or VS Code.
2.  Run **Cell 1** to load the environment and initialize the Auditor.
3.  Edit **Cell 2** to set your test question.
4.  Run **Cell 3** to see what context is retrieved from your PDFs.
5.  Run **Cell 4** to generate an answer and see the self-critique/validation score.

## Running a Full Audit

To run the auditor on the entire CSV input:

```bash
python rcm_engine.py
```

Results will be saved to `audit_results.json` (or the path configured in `config.yaml`).
