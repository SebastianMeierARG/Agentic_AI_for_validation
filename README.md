# Audit Tool for IFRS 9 Validation

This tool automates the auditing process by analyzing design effectiveness assessments against provided documentation using RAG (Retrieval-Augmented Generation).

https://mermaid.ai/app/projects/d845e351-9519-438c-8681-d427564ff745/diagrams/6a1d2297-eb75-4b90-a3cb-4695b9af63eb/version/v0.1/edit

## Features
- **Persistent Vector Store**: Eficiently loads/saves document embeddings.
- **Dual-Memory RAG**: Queries both Client Documents and Regulations (if available).
- **Compliance Verdict**: Automatically classifies findings (Compliant, Non-Compliant, etc.).
- **Client Summary**: Generates high-level summaries of client policies.

## High-Level Flow
At its core, the tool operates as a **Retrieval-Augmented Generation (RAG)** pipeline designed to automate compliance auditing (specifically for IFRS 9). The flow works like this:
1. **Ingest Documents:** It reads the regulatory policies and methodology documents you provide (`documents/`).
2. **Read Audit Questions:** It reads a list of audit controls and questions from an input CSV (`inputs/rcm_input.csv`).
3. **Retrieve Context:** For every question, the `rag_engine.py` searches the ingested documents for the most relevant paragraphs using semantic search.
4. **Generate Answer:** It sends the context and the question to an AI model using a strict prompt template (`templates/auditor_response.j2`) that forces it to verify facts before answering.
5. **Validate:** Optionally, `validate_audit.py` mathematically compares the AI's answers against a human expert's answers (`inputs/rcm_expert_answer.csv`) using NLP metrics (Cosine and Jaccard) to score the AI's accuracy.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Variables**:
    Create a `.env` file with your API keys:
    ```
    OPENAI_API_KEY=your_key_here
    GOOGLE_API_KEY=your_key_here
    ```
3.  **Configuration**:
    Edit `config.yaml` to set your preferred model provider (`openai` or `google`) and other settings.

## Usage

### 1. Prepare Documents
- Place client PDF documents in the `documents/` folder.
- (Optional) Place regulation PDF documents in the `regulations/` folder.

### 2. Prepare Input
- Ensure `inputs/rcm_input.csv` contains the audit control references and questions.
- Ensure `inputs/rcm_expert_answer.csv` exists for validation.

### 3. Run the Audit
Run the main script:
```bash
python src/run_audit.py
```
This will:
- Build/Load the vector indices (Client + Regulations).
- Process each row in `inputs/rcm_input.csv`.
- Generate answers and compliance verdicts.
- Save results to `outputs/audit_results.json`.

### 4. Generate Client Summary
To generate a standalone summary of the client's policies:
```bash
python -c "import sys; sys.path.append('src'); from rcm_engine import RcmAuditor; RcmAuditor().generate_client_summary()"
```
Output will be saved to `outputs/client_summary.txt`.

### 5. Validate Results (Expert Comparison)
To programmatically compare AI answers against expert ground truth using deterministic NLP metrics (Semantic Cosine Similarity via `sentence-transformers` and Factual Jaccard Overlap):
```bash
python src/validate_audit.py
```
Output: `outputs/validation_comparison_report.csv`.

### 6. Interactive Testing
Open `notebooks/interactive_audit.ipynb` in Jupyter. The notebook automatically adds `../src` to the path.

## Output
- **`outputs/audit_results.json`**: Detailed audit findings.
- **`outputs/validation_comparison_report.csv`**: Comparison vs expert answers.

## Folder Structure
- `src/`: Core Python scripts (`rcm_engine.py`, `rag_engine.py`, etc.).
- `notebooks/`: Jupyter notebooks (`interactive_audit.ipynb`).
- `inputs/`: Input CSVs (`rcm_input.csv`, `rcm_expert_answer.csv`).
- `outputs/`: Generated results.
- `documents/`: Client PDFs.
- `regulations/`: Regulation PDFs.
- `faiss_index_client/`, `faiss_index_regs/`: Persistent vector indices.
- `old_scripts/`: Archived verification scripts.
