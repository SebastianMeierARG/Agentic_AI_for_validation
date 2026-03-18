# Audit Tool for IFRS 9 Validation

This tool automates the auditing process by analyzing design effectiveness assessments against provided documentation using RAG (Retrieval-Augmented Generation).

https://mermaid.ai/app/projects/d845e351-9519-438c-8681-d427564ff745/diagrams/6a1d2297-eb75-4b90-a3cb-4695b9af63eb/version/v0.1/edit

## Features
- **Persistent Vector Store**: Efficiently loads/saves document embeddings.
- **Dual-Memory RAG**: Queries both Client Documents and Regulations (if available).
- **Score-Threshold + CrossEncoder Reranking**: Filters low-relevance chunks before they reach the LLM, then reranks survivors with a CrossEncoder for higher precision.
- **Compliance Verdict**: Automatically classifies findings (Compliant, Non-Compliant, Partial, Insufficient Info).
- **Three-Tier Check Classification**: Audit procedures are classified as `DOCUMENTATION_CHECK`, `METHODOLOGY_CHECK`, or `QUANTITATIVE_CHECK`, with progressively stricter scrutiny applied for quantitative parameters (PD, LGD, SICR, FLI, ECL).
- **Client Summary**: Generates high-level summaries of client policies across 13 IFRS 9 governance topics.
- **Tier Filtering**: Enables filtering of audit processing and validation by tier criteria.
- **Risk-Weighted Compliance Scoring**: Validation report computes both unweighted and tier-weighted accuracy scores.
- **Audit Trail**: Every run produces a timestamped archive copy, a SHA-256 document hash manifest, and a `flagged_for_review.json` for human escalation.
- **Prompt Injection Sanitization**: Retrieved document chunks are scanned and redacted for common injection patterns before being passed to the LLM.

## Validation Mechanisms
The tool incorporates a multi-layered approach to ensure reliability and factual accuracy:

### 1. Extrinsic Validation (Expert Comparison)
Located in `validate_audit.py`, this mechanism compares AI-generated answers against a human expert's ground truth (`inputs/rcm_expert_answer.csv`).
- **CrossEncoder Semantic Similarity**: Uses `cross-encoder/stsb-roberta-base` to evaluate conceptual overlap between AI and expert answers.
- **LLM-as-a-Judge**: A secondary LLM evaluates accuracy, stability, drift resistance, and guardrail effectiveness, each scored 0–100.
- **Risk-Weighted Score**: Scores are multiplied by tier weights (Tier 1 = 3×, Tier 2 = 2×, Tier 3 = 1×) and aggregated into an overall risk-weighted accuracy score.
- Outputs `outputs/validation_comparison_report.csv` with per-control scores, weights, and LLM reasoning.

### 2. Intrinsic Validation (Self-Critique QA Step)
Located in `rcm_engine.py` (using `auditor_critique.j2`):
- After the AI generates an answer, the same LLM re-evaluates it against the raw context.
- **Scoring (0–10)**: Evaluates Truthfulness and Thoroughness.
- **Hallucination Checks**: Calculates `hallucination_rate` and `hallucination_count`; severely penalises invented facts (Score: 0).

### 3. Cross-LLM Hallucination Check
- An independent open-source LLM (**Llama 3.3 70B via Groq**) reviews the generated answer against the context, completely separate from the primary LLM provider.
- Flags any claims not supported by the retrieved documents.
- Results stored as `Cross_LLM_Hallucinated` and `Cross_LLM_Concerns` fields in the audit output.
- Requires `GROQ_API_KEY` in `.env` (free tier available at console.groq.com). Falls back to the secondary API provider, then the primary LLM, if Groq is unavailable.

### 4. Confidence Score
- Each control receives a `Confidence_Score` (0–100) computed as the average of the normalised self-critique score and the cross-LLM confidence rating.
- If the cross-LLM detects hallucination, a 30-point penalty is applied to that side of the average.
- Controls below the configurable `confidence_threshold` (default 60) are automatically flagged for human review.

### 5. Prompt-Level Grounding & Retrieval Constraints
- **Strict Evidence Citations**: `auditor_response.j2` enforces page and document citations after every fact.
- **QUANTITATIVE_CHECK Probes**: For numerical parameters the prompt explicitly requires: a quantitative PD lifetime threshold for SICR (DPD-only is flagged); ≥3 probability-weighted macro scenarios for FLI; backtesting results with confidence intervals for PD/LGD/EAD.
- **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical answer in the document language to improve semantic search across language boundaries (Spanish docs, English queries).
- **Score Threshold Filtering**: Chunks with L2 distance above `retrieval_score_threshold` are discarded before reaching the LLM.
- **CrossEncoder Reranking**: Surviving chunks are reranked and trimmed to `rerank_top_k` before prompt construction.
- **Prompt Injection Sanitization**: Chunks are scanned for injection patterns (e.g., `IGNORE PREVIOUS INSTRUCTIONS`) and redacted before use.

## High-Level Flow
1. **Ingest Documents**: Reads PDFs from `documents/` and optionally `regulations/`.
2. **Read Audit Questions**: Reads controls and test procedures from `inputs/rcm_input.csv`.
3. **Retrieve Context**: For each question, `rag_engine.py` runs HyDE → similarity search with score filtering → CrossEncoder reranking.
4. **Generate Answer**: Sends reranked context to the LLM via `auditor_response.j2`, which enforces classification (`DOCUMENTATION_CHECK` / `METHODOLOGY_CHECK` / `QUANTITATIVE_CHECK`) and citation.
5. **Self-Critique**: `auditor_critique.j2` scores the answer 0–10 for truthfulness and thoroughness.
6. **Cross-LLM Check**: Secondary provider independently checks for unsupported claims.
7. **Confidence Score**: Blended confidence (0–100) computed from steps 5 and 6.
8. **Audit Trail**: Timestamped copy, SHA-256 document hashes, and run manifest saved; flagged controls written to `flagged_for_review.json`.
9. **Validate** (optional): `validate_audit.py` compares AI answers to expert ground truth using CrossEncoder + LLM-as-a-judge, with risk-weighted scoring.

## Setup

1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2. **Environment Variables**:
    Create a `.env` file with your API keys:
    ```
    OPENAI_API_KEY=your_key_here
    GROQ_API_KEY=your_key_here     # Free at console.groq.com — used for LLM-as-a-judge
    GOOGLE_API_KEY=your_key_here   # Optional fallback if GROQ_API_KEY is not set
    ```
3. **Configuration**:
    Edit `config.yaml` to set your preferred model provider (`openai` or `google`) and tune the following key settings:

    | Setting | Default | Description |
    |---|---|---|
    | `llm_settings.provider` | `openai` | Primary LLM provider |
    | `rag_settings.retrieval_score_threshold` | `1.2` | Max L2 distance for chunk retention |
    | `rag_settings.rerank_top_k` | `6` | Chunks kept after reranking |
    | `rag_settings.reranker_model` | `cross-encoder/stsb-roberta-base` | Swap for multilingual model in production |
    | `judge_llm.model` | `llama-3.3-70b-versatile` | Open model used for judging (via Groq) |
    | `validation.enable_cross_llm_critique` | `true` | Requires `GROQ_API_KEY` (or fallback key) |
    | `validation.confidence_threshold` | `60.0` | Below this, control is flagged |
    | `audit_trail.flag_score_threshold` | `6` | Self-critique score below this flags the control |
    | `filtering.tier_weights` | `{1:3, 2:2, 3:1}` | Used for risk-weighted validation scoring |

## Usage

### 1. Prepare Documents
- Place client PDF documents in the `documents/` folder.
- (Optional) Place regulation PDF documents in the `regulations/` folder.

### 2. Prepare Input
- Ensure `inputs/rcm_input.csv` contains the audit control references and questions.
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
This will:
- Build/load the vector indices (Client + Regulations).
- Process each row in `inputs/rcm_input.csv` with HyDE → filtering → reranking → LLM generation → self-critique → cross-LLM check.
- Save results to `outputs/audit_results.json`.
- Save a timestamped archive to `outputs/run_history/`.
- Write `outputs/run_manifest.json` (run ID, UTC timestamp, LLM model, SHA-256 document hashes).
- Write `outputs/flagged_for_review.json` with any controls below confidence/score thresholds.

### 5. Validate Results (Expert Comparison)
```bash
python src/validate_audit.py
```
Output: `outputs/validation_comparison_report.csv` with CrossEncoder scores, LLM judge scores (accuracy, stability, drift resistance, guardrails), tier weights, and overall risk-weighted accuracy.

### 6. Interactive Dashboard
```bash
shiny run shiny_app.py
```
The dashboard includes:
- **Control Center**: Upload PDFs and CSVs; trigger summary generation, audit, and validation.
- **Client Summary**: View or regenerate the AI policy summary in formatted markdown.
- **Audit Findings**: Filter by scope, verdict, and tier; yellow banner alerts when controls are flagged for human review; row-level detail view for AI answer and evidence sources.
- **Validation Report**: KPI cards (colour-coded by score threshold) showing Cross-Encoder, LLM Accuracy, Stability, Drift Resistance, and Guardrail averages; row-level reasoning and expert answer detail.

### 7. Interactive Testing
Open `notebooks/interactive_audit.ipynb` in Jupyter for step-by-step decoupled validation workflow.

## Output Files

| File | Description |
|---|---|
| `outputs/audit_results.json` | Detailed audit findings including AI answer, verdict, self-critique score, confidence score, and cross-LLM hallucination flags |
| `outputs/flagged_for_review.json` | Controls requiring human review with specific flag reasons |
| `outputs/run_manifest.json` | Run metadata: timestamp, model, SHA-256 document hashes, total controls processed |
| `outputs/run_history/` | Immutable timestamped copies of each audit run |
| `outputs/client_summary.md` | AI-generated summary of client's IFRS 9 policy stance |
| `outputs/validation_comparison_report.csv` | Expert comparison with CrossEncoder scores, LLM judge scores, tier weights, and risk-weighted accuracy |

## Folder Structure
- `shiny_app.py`: Interactive validation monitoring dashboard.
- `src/`: Core Python scripts (`rcm_engine.py`, `rag_engine.py`, `validate_audit.py`, `run_audit.py`, `run_summary.py`, etc.).
- `templates/`: Jinja2 prompt templates (`auditor_response.j2`, `auditor_critique.j2`, `client_summary.j2`).
- `notebooks/`: Jupyter notebooks (`interactive_audit.ipynb`).
- `inputs/`: Input CSVs (`rcm_input.csv`, `rcm_expert_answer.csv`).
- `outputs/`: Generated results and audit trail.
- `documents/`: Client PDFs.
- `regulations/`: Regulation PDFs.
- `faiss_index_client/`, `faiss_index_regs/`: Persistent vector indices.



