# IFRS 9 Agentic Audit Validation Tool — Claude Code Guide

## Project Overview
RAG + LLM pipeline for automated IFRS 9 / ECL audit compliance validation.
Processes a Risk and Control Matrix (RCM) CSV, retrieves evidence from client PDFs and regulatory documents, generates structured audit answers, and validates them through a multi-layer quality framework.

Client documents are in **Spanish**; regulations are in **English**.

---

## Folder Structure

```
Agentic_AI_for_validation/
├── src/                        # All Python source modules
├── templates/                  # Jinja2 prompt templates
├── notebooks/                  # Jupyter notebooks (interactive debug)
├── inputs/                     # Input CSVs
├── documents/                  # Client PDFs (Spanish) ingested into RAG
├── regulations/                # Regulatory PDFs (IFRS 9 / EBA guidelines)
├── outputs/                    # Per-run timestamped folders (see below)
├── faiss_index_client/         # Persistent FAISS vector store — client docs
├── faiss_index_regs/           # Persistent FAISS vector store — regulations
├── config.yaml                 # All runtime configuration
├── requirements.txt            # Python dependencies
└── .env                        # API keys (never commit)
```

---

## Source Modules (`src/`)

| File | Role |
|---|---|
| `config.py` | Loads `config.yaml` + `.env`; resolves all paths to absolute |
| `llm_factory.py` | Returns primary LLM, embeddings, and judge LLM with full fallback chain |
| `rag_engine.py` | Builds/loads FAISS indices; HyDE retrieval; multilingual CrossEncoder reranking |
| `rcm_engine.py` | Core audit logic per row: retrieval → answer → self-critique → cross-LLM check |
| `run_audit.py` | Entry point: reads RCM CSV, loops rows, saves all outputs, calls `validate_audit` |
| `validate_audit.py` | Compares AI answers to expert ground truth; CrossEncoder + LLM-as-judge scoring |
| `run_summary.py` | Generates a markdown client policy summary across 13 IFRS 9 governance topics |
| `generate_performance_report.py` | Parses `val_metrics_*.csv`; produces markdown performance report |
| `pipeline.py` | High-level orchestration wrapper |

---

## Configuration (`config.yaml`)

```yaml
llm_settings:
  provider: "openai"            # "openai" or "google"
  temperature: 0.0
  openai:
    model: "gpt-4o-mini"
    embedding_model: "text-embedding-3-small"
  google:
    model: "models/gemini-pro-latest"
    embedding_model: "models/embedding-001"

rag_settings:
  chunk_size: 1200
  chunk_overlap: 300
  document_language: "Spanish"
  retrieval_score_threshold: 1.8   # L2 — permissive for cross-lingual embeddings
  use_regulations: true
  client_top_k: 6                  # max client chunks per query
  regs_top_k: 2                    # capped to prevent crowding out client docs
  rerank_top_k: 7                  # final chunks passed to LLM after reranking
  reranker_model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

judge_llm:
  # Fallback chain (first available wins):
  #   1. Ollama (local, zero cost, zero quota) — requires Ollama running locally
  #   2. Groq   (free cloud, fast)             — requires GROQ_API_KEY
  #   3. Together AI (free model)              — requires TOGETHER_API_KEY
  #   4. Secondary provider (Google/OpenAI)
  #   5. Primary LLM — last resort
  model: "llama-3.3-70b-versatile"
  together_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
  ollama_model: "llama3.1:8b"
  ollama_base_url: "http://localhost:11434"

validation:
  enable_self_critique: true        # judge LLM scores the answer 0-10
  enable_cross_llm_critique: true   # independent LLM checks for hallucinations
  confidence_threshold: 60.0        # below this → flagged for human review

audit_trail:
  enabled: true
  flag_score_threshold: 6           # self-critique score < 6 → flagged

filtering:
  tier: '1'                         # 'all' | '1' | '2' | '3'
  tier_weights:
    '1': 3
    '2': 2
    '3': 1

paths:
  input_csv: "inputs/rcm_input.csv"
  output_json: "outputs/audit_results.json"         # fallback for notebook only
  documents_folder: "documents/"
  expert_answers_csv: "inputs/rcm_expert_answer.csv"
  validation_report_csv: "outputs/val_metrics.csv"  # fallback for notebook only
```

> `output_json` and `validation_report_csv` are fallbacks used only by the notebook.
> `run_audit.py` saves everything into a timestamped run folder.

---

## Output Structure

Each full run creates:
```
outputs/
  20260320T143022Z/
    audit_results.json            # AI answers, verdicts, scores for every control
    run_manifest.json             # run_id, config snapshot, SHA-256 doc hashes, LLM used
    flagged_for_review.json       # controls below confidence/score thresholds
    val_metrics_<ts>_<judge>.csv  # validation report vs expert answers
```

### audit_results.json — per row
```json
{
  "Control Reference": "1.1",
  "AI_Answer": "Yes... [Page 3 of 'Policy.pdf']",
  "Evidence_Sources": "Page 1-2 of 'Policy.pdf'",
  "Compliance_Verdict": "Compliant",
  "Validation_Score": 8,
  "Validation_Reasoning": "...",
  "Hallucination_Rate": 0.0,
  "Confidence_Score": 87.5,
  "Cross_LLM_Hallucinated": false,
  "Cross_LLM_Concerns": ""
}
```

### val_metrics_*.csv — key columns
| Column | Description |
|---|---|
| `Cross_Encoder_Score` | Semantic similarity vs expert answer (0–100) |
| `Validation_Score_LLM_as_judge` | LLM accuracy rating vs expert answer (0–100) |
| `Stability_Score` | Logical consistency rating (0–100) |
| `Drift_Resistance_Score` | Grounding in context, no hallucination drift (0–100) |
| `Guardrail_Effectiveness_Score` | Professional tone rating (0–100) |
| `Reasoning_LLM_as_judge` | Explanation of scores |
| `Weighted_Accuracy_Score` | Tier-weighted accuracy for overall metric |

---

## Execution Entry Points

```bash
# Full pipeline: audit + validation in one command
python src/run_audit.py

# Validation only (auto-detects latest run folder)
python src/validate_audit.py

# Client policy summary (one-time)
python src/run_summary.py

# Performance report from validation CSV
python src/generate_performance_report.py

# Interactive debug
jupyter notebook notebooks/interactive_audit.ipynb
```

---

## Audit Pipeline (per control row)

```
1. HyDE retrieval
   ├─ Generate hypothetical answer in document_language (Spanish)
   ├─ FAISS search: client (top_k=6) + regulations (top_k=2) separately
   ├─ Filter by L2 threshold (1.8)
   └─ Rerank combined results with multilingual CrossEncoder (top_k=7)

2. Answer generation
   ├─ Render auditor_response.j2 with context
   ├─ Classify: DOCUMENTATION_CHECK | METHODOLOGY_CHECK | QUANTITATIVE_CHECK
   ├─ Cite every fact as [Page X of 'Filename']
   └─ Assign Compliance_Verdict: Compliant | Non-Compliant | Partial | Insufficient Info

3. Self-critique  (if enable_self_critique: true)
   ├─ Independent judge LLM (Ollama-first) scores answer via auditor_critique.j2
   └─ Returns score (0–10), reasoning, hallucination_rate

4. Cross-LLM hallucination check  (if enable_cross_llm_critique: true)
   ├─ Independent judge LLM checks for unsupported claims
   └─ Returns hallucinated (bool) + concerns

5. Confidence score
   └─ Blends self-critique score + cross-LLM confidence → 0–100
```

---

## Validation Architecture (4 Layers)

| Layer | Who validates | What it checks | Output fields |
|---|---|---|---|
| **Self-critique** | Judge LLM (Ollama-first) reviewing primary answer | Truthfulness, thoroughness, hallucinations | `Validation_Score`, `Validation_Reasoning`, `Hallucination_Rate` |
| **Cross-LLM** | Independent judge LLM | Unsupported factual claims | `Cross_LLM_Hallucinated`, `Cross_LLM_Concerns` |
| **Expert comparison** | CrossEncoder (no LLM) | Semantic similarity vs expert answer | `Cross_Encoder_Score` |
| **LLM-as-judge** | Judge LLM vs expert answer | Accuracy, Stability, Drift, Guardrail | `Validation_Score_LLM_as_judge` + 3 sub-scores |

> Self-critique intentionally uses `get_judge_llm()` (Ollama/independent), NOT the primary LLM,
> to avoid the model grading its own output.

---

## Judge LLM Fallback Chain

Defined in `llm_factory.py`. Order (first available wins):

1. **Ollama** — `llama3.1:8b` local, zero cost, zero quota. Requires Ollama running (`ollama serve`).
2. **Groq** — `llama-3.3-70b-versatile`, free cloud. Requires `GROQ_API_KEY`. 100k tokens/day limit.
3. **Together AI** — `Llama-3.3-70B-Instruct-Turbo-Free`, free cloud. Requires `TOGETHER_API_KEY`.
4. **Secondary provider** — Google if primary is OpenAI, or vice versa.
5. **Primary LLM** — last resort; least independent.

On quota exhaustion (429 TPD), credit limit (402), or auth failure (401), the system automatically advances to the next provider with no manual intervention.

---

## Prompt Templates (`templates/`)

| Template | Used by | Purpose |
|---|---|---|
| `auditor_response.j2` | `rcm_engine.py` | Main audit answer: classification, evidence citation, verdict |
| `auditor_critique.j2` | `rcm_engine.py` | Self-critique scoring (0–10) + hallucination detection |
| `client_summary.j2` | `run_summary.py` | Extract 13 IFRS 9 governance topics from client docs |

---

## Notebook (`notebooks/interactive_audit.ipynb`)

Sections:
| # | Section | Tokens/row | Purpose |
|---|---|---|---|
| 0 | Setup | — | Load env, init RAG, set LITE_MODE |
| 1 | RAG Debug | ~300 | Inspect retrieved chunks before spending tokens |
| 2 | Single Control — LITE | ~800 | Answer only, no critique |
| 3 | Single Control — FULL | ~2500 | Answer + self-critique + cross-LLM |
| 3b | Custom Question | ~800 | Ad-hoc query without CSV |
| 4 | Compare vs Expert | ~600 | CrossEncoder + optional LLM judge |
| 5 | Adversarial Test | ~800–2500 | Edge case / trap questions |
| 6 | Batch (N rows) | N × mode | Mini audit run |
| 7 | Full Validation | N × judge | Runs `validate_audit()` on latest run |
| 8 | Ollama Test | — | Smoke-test Ollama connection and model |

**LITE_MODE = True** (default in Section 0) skips self-critique and cross-LLM (~3× fewer tokens).

---

## Ollama Setup

```bash
# Install from https://ollama.com, then:
ollama pull llama3.1:8b     # ~5GB download, recommended model

# Verify
ollama list
```

Ollama runs as a background service on `http://localhost:11434`.
No API key required. No token quota. Inference speed depends on hardware (GPU >> CPU).

---

## Key Design Decisions

- **Source-balanced retrieval:** Client and regulation indices queried separately with independent caps to prevent either from dominating context.
- **Permissive L2 threshold (1.8):** Cross-lingual embeddings (Spanish docs × English HyDE query) produce higher L2 distances than monolingual pairs; threshold tuned to avoid filtering valid client content.
- **Multilingual reranker:** `mmarco-mMiniLMv2-L12-H384-v1` handles Spanish/English mixed pairs correctly.
- **Prompt injection sanitization:** Retrieved chunks scanned for injection patterns before LLM ingestion.
- **Three-tier check classification:** QUANTITATIVE_CHECK applies maximum scrutiny; DOCUMENTATION_CHECK is least strict.
- **Timestamped run folders:** SHA-256 document hashes in `run_manifest.json` allow proving document integrity to regulators after the fact.
- **Risk-weighted scoring:** Tier 1 controls weighted 3×, Tier 2 2×, Tier 3 1× in the final accuracy metric.

---

## API Keys Required (`.env`)

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
GROQ_API_KEY=...
TOGETHER_API_KEY=...
```

Ollama requires no API key.
