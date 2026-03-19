# Audit Tool for IFRS 9 Validation

This tool automates the auditing process by analysing design effectiveness assessments against provided documentation using RAG (Retrieval-Augmented Generation).

## Features
- **Persistent Vector Store**: Efficiently loads/saves document embeddings via FAISS.
- **Dual-Memory RAG**: Queries both Client Documents and Regulations independently with source-balanced retrieval caps.
- **Score-Threshold + CrossEncoder Reranking**: Filters low-relevance chunks before the LLM, then reranks survivors with a multilingual CrossEncoder for higher precision.
- **Language-Independent Pipeline**: Set `document_language` in `config.yaml` to any language (Spanish, English, German, French…); HyDE, reranking, and translation adapt automatically.
- **Compliance Verdict**: Automatically classifies findings as Compliant, Non-Compliant, Partial, or Insufficient Info.
- **Three-Tier Check Classification**: Audit procedures are classified as `DOCUMENTATION_CHECK`, `METHODOLOGY_CHECK`, or `QUANTITATIVE_CHECK`, with progressively stricter scrutiny for quantitative parameters (PD, LGD, SICR, FLI, ECL).
- **Multi-Provider Judge LLM Fallback**: Cross-LLM judging uses a cascading fallback chain — Groq → Together AI → Ollama (local) → secondary provider → primary LLM — with automatic provider switching on quota exhaustion (429 TPD), credit limits (402), or auth failures (401).
- **Client Summary**: Generates high-level summaries of client policies across 13 IFRS 9 governance topics.
- **Tier Filtering & Risk-Weighted Scoring**: Filter audit processing by tier; validation report computes both unweighted and tier-weighted accuracy scores.
- **Audit Trail**: Every run produces a timestamped archive copy, SHA-256 document hashes, a `run_manifest.json` (including full config snapshot), and `flagged_for_review.json` for human escalation.
- **Prompt Injection Sanitisation**: Retrieved document chunks are scanned and redacted for common injection patterns before being passed to the LLM.
- **Traceable Validation Reports**: Each validation run saves a CSV with the judge model label and UTC timestamp embedded in the filename (e.g. `val_metrics_20260318T131118Z_groq_llama-3.3-70b-versatile.csv`).

## Validation Mechanisms

### 1. Extrinsic Validation (Expert Comparison)
Located in `validate_audit.py`, compares AI-generated answers against a human expert's ground truth (`inputs/rcm_expert_answer.csv`).
- **CrossEncoder Semantic Similarity**: Uses `cross-encoder/stsb-roberta-base` to evaluate conceptual overlap between AI and expert answers (0–100).
- **LLM-as-a-Judge**: An independent judge LLM evaluates accuracy, stability, drift resistance, and guardrail effectiveness, each scored 0–100.
- **Risk-Weighted Score**: Scores are multiplied by tier weights (Tier 1 = 3×, Tier 2 = 2×, Tier 3 = 1×) and aggregated into an overall risk-weighted accuracy score.
- Output filename includes the judge model used and UTC timestamp, so every validation run is traceable.

### 2. Intrinsic Validation (Self-Critique)
Located in `rcm_engine.py` (using `auditor_critique.j2`):
- After the AI generates an answer, the same LLM re-evaluates it against the raw context.
- **Scoring (0–10)**: Evaluates Truthfulness and Thoroughness.
- **Hallucination Checks**: Calculates `hallucination_rate`; severely penalises invented facts (Score: 0).

### 3. Cross-LLM Hallucination Check
An independent LLM reviews the generated answer against the retrieved context, completely separate from the primary LLM provider.
- **Fallback chain (first available wins):** Groq (Llama 3.3 70B) → Together AI (Llama 3.3 70B Free) → Ollama (local) → secondary provider (Google/OpenAI) → primary LLM.
- If a provider hits its quota (429 TPD), credit limit (402), or fails auth (401), the system switches automatically to the next provider mid-run — no zeros in the output.
- Results stored as `Cross_LLM_Hallucinated` and `Cross_LLM_Concerns` in the audit output.

### 4. Confidence Score
- Each control receives a `Confidence_Score` (0–100) blended from normalised self-critique and cross-LLM confidence.
- A 30-point penalty is applied if the cross-LLM detects hallucination.
- Controls below `confidence_threshold` (default 60) are automatically flagged for human review.

### 5. Prompt-Level Grounding & Retrieval Constraints
- **Strict Evidence Citations**: `auditor_response.j2` enforces page and document citations after every fact.
- **QUANTITATIVE_CHECK Probes**: Requires quantitative PD lifetime threshold for SICR; ≥3 probability-weighted macro scenarios for FLI; backtesting results with confidence intervals for PD/LGD/EAD.
- **HyDE**: Generates a hypothetical answer in `document_language` to improve cross-language semantic search.
- **Score Threshold Filtering + Multilingual Reranking**: Permissive L2 threshold (1.8) prevents valid non-English chunks from being discarded; multilingual `mmarco-mMiniLMv2-L12-H384-v1` reranker handles Spanish, German, French natively.
- **Prompt Injection Sanitisation**: Chunks are scanned for injection patterns and redacted before use.

## High-Level Flow
1. **Ingest Documents**: Reads PDFs from `documents/` and optionally `regulations/`.
2. **Read Audit Questions**: Reads controls and test procedures from `inputs/rcm_input.csv`.
3. **Retrieve Context**: For each question, `rag_engine.py` runs HyDE → source-balanced similarity search → score threshold filtering → multilingual CrossEncoder reranking.
4. **Generate Answer**: Sends reranked context to the LLM via `auditor_response.j2`, enforcing classification (`DOCUMENTATION_CHECK` / `METHODOLOGY_CHECK` / `QUANTITATIVE_CHECK`) and citation.
5. **Self-Critique**: `auditor_critique.j2` scores the answer 0–10 for truthfulness and thoroughness.
6. **Cross-LLM Check**: Independent judge LLM (Groq/Together AI/Ollama/fallback) checks for unsupported claims.
7. **Confidence Score**: Blended confidence (0–100) computed from steps 5 and 6.
8. **Audit Trail**: Timestamped copy, SHA-256 document hashes, config snapshot, and run manifest saved; flagged controls written to `flagged_for_review.json`.
9. **Validate** (optional): `validate_audit.py` compares AI answers to expert ground truth using CrossEncoder + LLM-as-a-judge, with risk-weighted scoring. Output CSV filename includes judge model label + UTC timestamp.

## Setup

1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2. **Environment Variables** — create a `.env` file:
    ```
    OPENAI_API_KEY=your_key_here
    GROQ_API_KEY=your_key_here           # Free at console.groq.com — primary judge LLM
    TOGETHER_API_KEY=your_key_here       # Free model at console.together.ai — fallback judge
    GOOGLE_API_KEY=your_key_here         # Optional — Google/Gemini fallback (also accepts GEMINI_API_KEY)
    ```
3. **Configuration** — edit `config.yaml`:

    | Setting | Default | Description |
    |---|---|---|
    | `llm_settings.provider` | `openai` | Primary LLM provider (`openai` or `google`) |
    | `llm_settings.temperature` | `0.0` | LLM temperature (0.0 = deterministic) |
    | `rag_settings.document_language` | `Spanish` | Language of client docs — any language supported |
    | `rag_settings.chunk_size` | `1200` | Token size for document chunks |
    | `rag_settings.chunk_overlap` | `300` | Overlap between adjacent chunks |
    | `rag_settings.retrieval_score_threshold` | `1.8` | Max L2 distance for chunk retention |
    | `rag_settings.use_regulations` | `true` | Set to `false` to skip the regulations index (saves ~1500 tokens/row) |
    | `rag_settings.client_top_k` | `6` | Max client document chunks per query |
    | `rag_settings.regs_top_k` | `2` | Max regulation chunks (capped to avoid crowding client docs) |
    | `rag_settings.rerank_top_k` | `7` | Final chunks passed to LLM after reranking |
    | `rag_settings.reranker_model` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual reranker |
    | `judge_llm.model` | `llama-3.3-70b-versatile` | Groq model for judging |
    | `judge_llm.together_model` | `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` | Together AI fallback model |
    | `judge_llm.ollama_model` | `llama3.2` | Local Ollama fallback model |
    | `validation.enable_self_critique` | `false` | Run self-critique scoring after each answer |
    | `validation.enable_cross_llm_critique` | `true` | Run cross-LLM hallucination check |
    | `validation.confidence_threshold` | `60.0` | Below this, control is flagged for human review |
    | `audit_trail.flag_score_threshold` | `6` | Self-critique score below this flags the control |
    | `filtering.tier` | `1` | Tier to process (`1`, `2`, `3`, or `all`) |
    | `filtering.tier_weights` | `{1:3, 2:2, 3:1}` | Multipliers for risk-weighted validation scoring |

## Usage

### 1. Prepare Documents
- Place client PDF documents in `documents/`.
- (Optional) Place regulation PDFs in `regulations/`.

### 2. Prepare Input
- Ensure `inputs/rcm_input.csv` contains audit control references and questions.
- Ensure `inputs/rcm_expert_answer.csv` exists for validation.

### 3. Generate Client Summary
```bash
python src/run_summary.py
```
Output saved to `outputs/client_summary.md`. Also available via the **Client Summary** tab in the dashboard.

### 4. Run the Audit
```bash
python src/run_audit.py
```
Builds/loads vector indices → processes each RCM row → saves results to `outputs/audit_results.json` → writes timestamped archive, run manifest (with config snapshot), and flagged controls.

### 5. Validate Results
```bash
python src/validate_audit.py
```
Output: `outputs/val_metrics_{timestamp}_{judge_model}.csv` with CrossEncoder scores, LLM judge scores (accuracy, stability, drift resistance, guardrails), tier weights, and risk-weighted accuracy.

### 5a. Full Pipeline (audit → validate in one command)
```bash
python src/pipeline.py
```
Runs `run_audit.py` then `validate_audit.py` sequentially.

### 5b. Generate Performance Report
```bash
python src/generate_performance_report.py
```
Reads a validation CSV and produces a Markdown report breaking down scores by scope, highlighting high and low performers across all validation dimensions.

### 6. Interactive Dashboard
```bash
shiny run shiny_app.py
```
- **Control Center**: Upload PDFs and CSVs; trigger summary, audit, and validation.
- **Client Summary**: View or regenerate the AI policy summary.
- **Audit Findings**: Filter by scope, verdict, and tier; yellow banner for flagged controls; row-level AI answer and evidence detail.
- **Validation Report**: KPI cards (colour-coded) for Cross-Encoder, LLM Accuracy, Stability, Drift Resistance, Guardrail averages; row-level reasoning and expert answer.

### 7. Interactive Debug Notebook
```bash
jupyter notebook notebooks/interactive_audit.ipynb
```
Row-by-row testing without running the full pipeline. **LITE mode** (default) skips self-critique and cross-LLM checks (~800 tokens/row vs ~2500 in FULL mode). Sections: RAG chunk inspector → single control test → custom question → expert comparison → adversarial test → batch run.

## Output Files

| File | Description |
|---|---|
| `outputs/audit_results.json` | AI answers, verdicts, self-critique scores, confidence scores, cross-LLM hallucination flags |
| `outputs/flagged_for_review.json` | Controls requiring human review with specific flag reasons |
| `outputs/run_manifest.json` | Run metadata: timestamp, model, SHA-256 document hashes, total controls, full config snapshot |
| `outputs/run_history/` | Immutable timestamped copies of each audit run |
| `outputs/client_summary.md` | AI-generated summary of client's IFRS 9 policy stance |
| `outputs/val_metrics_{ts}_{judge}.csv` | Expert comparison: CrossEncoder scores, LLM judge scores, tier weights, risk-weighted accuracy |

## Folder Structure
- `shiny_app.py`: Interactive validation monitoring dashboard.
- `src/`: Core scripts (`rcm_engine.py`, `rag_engine.py`, `validate_audit.py`, `run_audit.py`, `llm_factory.py`, `pipeline.py`, `generate_performance_report.py`, etc.).
- `templates/`: Jinja2 prompt templates (`auditor_response.j2`, `auditor_critique.j2`, `client_summary.j2`).
- `notebooks/interactive_audit.ipynb`: Row-by-row debug notebook with LITE/FULL mode toggle.
- `inputs/`: Input CSVs (`rcm_input.csv`, `rcm_expert_answer.csv`).
- `outputs/`: Generated results and audit trail.
- `documents/`: Client PDFs (any language).
- `regulations/`: Regulation PDFs.
- `faiss_index_client/`, `faiss_index_regs/`: Persistent FAISS vector indices.
