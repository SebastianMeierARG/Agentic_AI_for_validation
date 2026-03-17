# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY, GEMINI_API_KEY
```

### Running the Pipeline
```bash
# Run the full audit (generates audit_results.json)
python src/run_audit.py

# Run expert comparison validation
python src/validate_audit.py

# Run full pipeline: audit → validate → report
python src/pipeline.py

# Generate performance report
python src/generate_performance_report.py

# Launch interactive Shiny dashboard
shiny run shiny_app.py
```

## Architecture

This is a RAG + LLM pipeline for IFRS 9 compliance auditing. It reads audit control questions from a CSV, retrieves relevant passages from client PDFs using FAISS vector search, and generates LLM answers with citations.

### Core Data Flow
```
inputs/rcm_input.csv (audit questions)
    ↓
src/run_audit.py (orchestrator)
    ↓
src/rag_engine.py → faiss_index_client/ + faiss_index_regs/ (FAISS retrieval)
    ↓
src/rcm_engine.py + templates/auditor_response.j2 (LLM answer generation)
    ↓
templates/auditor_critique.j2 (intrinsic self-scoring 0-10)
    ↓
outputs/audit_results.json
    ↓
src/validate_audit.py (cosine similarity + LLM-as-judge vs expert answers)
    ↓
outputs/validation_comparison_report.csv
```

### Key Components

**`src/config.py`** — Loads `config.yaml` and `.env`. All paths are resolved to absolute via `PROJECT_ROOT`. Controls LLM provider (OpenAI vs Gemini), RAG chunk size/overlap, document language (Spanish/English), and tier filtering.

**`src/llm_factory.py`** — Returns the correct `ChatOpenAI` or `ChatGoogleGenerativeAI` instance and corresponding embeddings based on config. Temperature is always 0.0.

**`src/rag_engine.py`** — Builds/loads persistent FAISS indices from PDFs in `documents/` and `regulations/`. Uses HyDE (Hypothetical Document Embeddings) to generate a synthetic answer in the target language before retrieval — this bridges Spanish/English language gaps. Implements batch embedding with exponential backoff for rate limits.

**`src/rcm_engine.py`** — Processes each audit row: combines Control Reference + Design Effectiveness + Test Procedures into a query, retrieves top-k chunks from both FAISS indices, renders Jinja2 templates, calls the LLM, extracts `<answer>`/`<evidence_sources>`/verdict, then runs self-critique scoring.

**`templates/auditor_response.j2`** — Main prompt template. Enforces two check types: `DOCUMENTATION_CHECK` (verify existence only) and `METHODOLOGY_CHECK` (require statistical justification). Every factual claim must be cited as `[Page X of 'Filename.ext']`.

**`templates/auditor_critique.j2`** — Self-critique prompt. Returns JSON with `score` (0-10), `hallucination_rate`, and `reasoning`. A score of 0 means hallucination; 10 means comprehensive and well-cited.

**`src/validate_audit.py`** — Loads `audit_results.json` and expert ground truth CSVs, computes cosine similarity via `sentence-transformers` (all-MiniLM-L6-v2), and runs an LLM-as-judge (0-100 scale) for each AI vs expert answer pair. Handles Spanish→English translation before embedding.

**`shiny_app.py`** — Interactive dashboard with three tabs: upload/run controls, audit findings explorer (filter by verdict/tier/score), and validation metrics. Streams terminal output in real-time.

### Configuration (`config.yaml`)
Key settings:
- `llm.provider`: `openai` or `google`
- `rag.chunk_size` / `rag.chunk_overlap`: Controls document chunking
- `rag.document_language`: `spanish` or `english` (affects HyDE generation language)
- `audit.tier`: `1`, `2`, `3`, or `all` — filters which RCM rows to audit
- `validation.enabled`: Toggle expert comparison step

### FAISS Indices
Indices in `faiss_index_client/` and `faiss_index_regs/` are rebuilt automatically if missing. They are tracked in git (modified state is normal after re-indexing). Place client PDFs in `documents/` and regulatory documents in `regulations/` before running.

### Output Format (`outputs/audit_results.json`)
Each entry contains:
- `control_reference`, `design_effectiveness`, `test_procedure`
- `answer` — LLM-generated answer with inline citations
- `evidence_sources` — list of `{filename, pages}` objects
- `verdict` — `Compliant` / `Non-Compliant` / `Partial` / `Insufficient Info`
- `critique_score` — intrinsic quality score 0-10
- `hallucination_rate` — float 0-1 from self-critique
