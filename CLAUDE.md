# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY, GEMINI_API_KEY
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

# Generate client summary only
python src/run_summary.py

# Launch interactive Shiny dashboard
shiny run shiny_app.py
```

## Architecture

RAG + LLM pipeline for IFRS 9 compliance auditing. Reads audit control questions from a CSV, retrieves relevant passages from client PDFs using FAISS vector search, generates LLM answers with citations, then runs multi-layer validation.

### Core Data Flow
```
inputs/rcm_input.csv
    ↓
src/run_audit.py (orchestrator)
    ↓
src/rag_engine.py → faiss_index_client/ + faiss_index_regs/
    (HyDE → source-balanced retrieval → score threshold → multilingual CrossEncoder reranking)
    ↓
src/rcm_engine.py + templates/auditor_response.j2 (LLM answer generation)
    ↓
templates/auditor_critique.j2 (self-scoring 0–10)
    ↓
cross-LLM critique via get_judge_llm() (hallucination check)
    ↓
outputs/audit_results.json + run_manifest.json + flagged_for_review.json
    ↓
src/validate_audit.py (CrossEncoder + LLM-as-judge vs expert answers)
    ↓
outputs/val_metrics_{timestamp}_{judge_model}.csv
```

### Key Components

**`src/config.py`** — Loads `config.yaml` and `.env`. All paths resolved to absolute via `PROJECT_ROOT`.

**`src/llm_factory.py`** — Central LLM factory. Key functions:
- `get_llm()` — returns primary LLM (OpenAI or Google) per config
- `get_embeddings()` — returns matching embeddings instance
- `get_judge_llm()` — cascading fallback: `_try_groq()` → `_try_together()` → `_try_ollama()` → `get_secondary_llm()` → `get_llm()`. Sets module-level `_judge_llm_label` string.
- `get_fallback_judge_llm()` — same chain but skips Groq (called after Groq TPD exhaustion)
- `get_judge_llm_label()` — returns filename-safe label of the judge LLM that was initialised

**`src/rag_engine.py`** — Builds/loads persistent FAISS indices. Key design:
- Two separate indices: `faiss_index_client/` and `faiss_index_regs/`
- `retrieve()`: HyDE query in `document_language` → separate similarity search per index with independent caps (`client_top_k`, `regs_top_k`) → score threshold filter → combined pool → multilingual CrossEncoder reranking → top `rerank_top_k` chunks returned
- L2 threshold default 1.8 (permissive) because cross-lingual embeddings produce higher distances than monolingual pairs
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual MS MARCO, ~270MB)
- Language is fully config-driven — changing `document_language` in `config.yaml` is sufficient

**`src/rcm_engine.py`** — Processes each audit row. Key methods:
- `process_row()` — orchestrates retrieve → sanitize → render → invoke → parse → self-critique → cross-LLM → confidence score
- `_sanitize_chunk()` — strips prompt injection patterns from retrieved chunks
- `_cross_llm_critique()` — calls judge LLM; has `_is_provider_exhausted()` static method that catches 401/402/429-TPD errors and walks through fallback providers
- `_invoke_with_retry()` — exponential backoff for per-minute rate limits

**`templates/auditor_response.j2`** — Main prompt. Three classification types: `DOCUMENTATION_CHECK` (existence only), `METHODOLOGY_CHECK` (require statistical justification), `QUANTITATIVE_CHECK` (maximum scrutiny: PD lifetime threshold for SICR, ≥3 weighted macro scenarios for FLI, backtesting for PD/LGD/EAD). Every fact cited as `[Page X of 'Filename.ext']`.

**`templates/auditor_critique.j2`** — Self-critique prompt. Returns JSON: `score` (0–10), `hallucination_rate` (0.0–1.0), `reasoning`.

**`src/validate_audit.py`** — Expert comparison. Key design:
- `_llm_box` + `_fallback_queue` pattern: starts with `get_judge_llm()`, maintains a queue `[get_fallback_judge_llm, get_secondary_llm, get_primary_llm]`. `_is_provider_exhausted()` catches 401/402/429-TPD; `_advance_provider()` pops next provider mid-run.
- Output path includes judge label + UTC timestamp: `val_metrics_{ts}_{judge_label}.csv`
- `_read_expert_csv()` tries 4 encodings: `utf-8 → utf-8-sig → windows-1252 → latin-1`
- `_safe_translate()` chunks long texts for Google Translate (4500 char limit)
- Risk-weighted scoring: `Tier_Weight` × `Validation_Score_LLM_as_judge`

**`src/run_audit.py`** — Orchestrator. Run manifest includes `config_snapshot` (full `config.yaml` parameters, JSON-serialised). Input CSV uses same 4-encoding fallback as validate_audit. SHA-256 hashes all PDFs in `documents/` and `regulations/`.

**`shiny_app.py`** — Dashboard with four tabs: Control Center, Client Summary, Audit Findings (flagged-controls banner), Validation Report (KPI cards + row detail).

### Configuration (`config.yaml`)
```yaml
llm_settings:
  provider: "openai"          # or "google"
  openai.model: "gpt-4o-mini"
  openai.embedding_model: "text-embedding-3-small"

rag_settings:
  document_language: "Spanish"   # any language — affects HyDE generation
  chunk_size: 1500
  chunk_overlap: 300
  retrieval_score_threshold: 1.8  # L2 distance cap; 1.8 suits cross-lingual embeddings
  client_top_k: 8
  regs_top_k: 4
  rerank_top_k: 10
  reranker_model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

judge_llm:
  model: "llama-3.3-70b-versatile"              # Groq
  together_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
  ollama_model: "llama3.2"
  ollama_base_url: "http://localhost:11434"

validation:
  enable_self_critique: true
  enable_cross_llm_critique: true
  confidence_threshold: 60.0

audit_trail:
  enabled: true
  flag_score_threshold: 6

filtering:
  tier: '1'
  tier_weights: {'1': 3, '2': 2, '3': 1}
```

### FAISS Indices
Rebuilt automatically if missing. Tracked in git (modified state is normal after re-indexing). Place client PDFs in `documents/` and regulatory documents in `regulations/` before running.

### Output Format (`outputs/audit_results.json`)
Each entry contains:
- `Control Reference`, `Design Effectiveness Assessment`, `Test Procedures`
- `AI_Answer` — answer with inline `[Page X of 'Filename.ext']` citations
- `Evidence_Sources` — pages and documents cited
- `Compliance_Verdict` — `Compliant` / `Non-Compliant` / `Partial` / `Insufficient Info`
- `Validation_Score` — self-critique 0–10
- `Hallucination_Rate` — float 0–1 from self-critique
- `Confidence_Score` — blended 0–100 from self-critique + cross-LLM
- `Cross_LLM_Hallucinated` — bool/null
- `Cross_LLM_Concerns` — string of flagged claims

### Validation Report Filename Pattern
`outputs/val_metrics_{YYYYMMDDTHHMMSSz}_{judge_label}.csv`

Example: `val_metrics_20260318T131118Z_groq_llama-3.3-70b-versatile.csv`

The judge label is set by `llm_factory._judge_llm_label` and reflects whichever provider actually ran (Groq, Together AI, Gemini, etc.).
