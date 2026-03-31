# User Manual — IFRS 9 Agentic Audit Validation Tool

**Version:** March 2026
**Audience:** Audit professionals, credit risk analysts, model validators

---

## Table of Contents

1. [What This Tool Does](#1-what-this-tool-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Before You Start — Know Your Resources](#3-before-you-start--know-your-resources)
4. [Validation Strategy A — Cross-LLM as Judge (Recommended for Regulatory Use)](#4-validation-strategy-a--cross-llm-as-judge)
5. [Validation Strategy B — Self-Critique Only (Faster, Less Compliant)](#5-validation-strategy-b--self-critique-only)
6. [Step-by-Step: Running a Full Audit](#6-step-by-step-running-a-full-audit)
7. [Understanding Your Results](#7-understanding-your-results)
8. [The Output Folder — What Each File Means](#8-the-output-folder--what-each-file-means)
9. [Configuring the Tool](#9-configuring-the-tool)
10. [Using the Interactive Notebook for Spot Checks](#10-using-the-interactive-notebook-for-spot-checks)
11. [Troubleshooting](#11-troubleshooting)
12. [Quick Reference Card](#12-quick-reference-card)
13. [Annex A — Architecture Diagram](#annex-a--architecture-diagram)
14. [Annex B — Technical Module Reference](#annex-b--technical-module-reference)

---

## 1. What This Tool Does

### The problem it solves

IFRS 9 is an international accounting standard that requires banks and financial institutions to estimate how much money they might lose on their loans — this estimate is called the **Expected Credit Loss (ECL)**. Regulators (such as the EBA in Europe or the BCRA in Argentina) periodically audit whether the institution's models, policies, and procedures actually comply with the standard.

These audits are time-consuming. An auditor must read dozens of internal policy documents, compare them against a checklist of controls (called a Risk and Control Matrix, or RCM), and write a documented conclusion for each control — complete with page references from the source documents.

This tool does that reading and drafting work automatically. Given your internal documents (PDFs) and a list of audit controls (a spreadsheet), it:

1. Reads every relevant page of your documents
2. For each control in the checklist, retrieves the most relevant passages
3. Drafts a structured answer — including which document page supports each claim
4. Classifies the control as **Compliant**, **Non-Compliant**, **Partial**, or **Insufficient Info**
5. Verifies that every page citation in the answer is grounded in the retrieved context
6. If the answer quality is low, automatically revises it using the critique as feedback
7. Has a second, independent AI model check every claim for accuracy
8. Flags any control where confidence is low, for a human auditor to review

### What IFRS 9 / ECL means in plain terms

The ECL can be interpreted as the bank's best estimate of "how much of what we lent out we probably won't get back." IFRS 9 requires this estimate to be calculated using a specific methodology, approved by senior governance, documented in formal policies, and regularly validated. The audit checks whether all of that is actually happening and properly evidenced.

### What the output looks like

For each control on your checklist, the tool produces:

- **Classification** — DOCUMENTATION_CHECK / METHODOLOGY_CHECK / QUANTITATIVE_CHECK
- **AI Answer** — A structured written response citing specific pages from your documents
- **Compliance Verdict** — Compliant / Non-Compliant / Partial / Insufficient Info
- **Confidence Score** — 0 to 100; decomposed into three sub-scores (retrieval, answer, citation)
- **Validation Score** — 0 to 10; how well the answer is supported by the documents
- **Citation Confidence** — % of cited page references verified against the retrieved context
- **Revision Count** — how many self-correction cycles were triggered for this control
- **Hallucination Flag** — whether the independent AI checker found any claims not backed by documents

---

## 2. Architecture Overview

This chapter explains how the tool works internally, in plain language. Understanding the architecture helps you interpret results, tune settings, and diagnose problems.

### The two-phase design

The tool is split into two sequential phases:

- **Phase 1 — Audit Generation:** Reads your documents, retrieves relevant passages, generates structured answers, and validates them through multiple quality layers.
- **Phase 2 — Validation:** Compares the AI answers to expert reference answers (if available) and scores them on accuracy, consistency, and hallucination resistance.

Phase 2 is optional. You can run Phase 1 alone (using `--no-validation`) and inspect raw results before deciding whether to validate.

---

### Phase 1 in depth: how an answer is produced

Each row in your audit checklist (`rcm_input.csv`) goes through the following steps:

#### Step 1 — Query Decomposition (optional)

Some audit controls are compound questions bundled into a single sentence — for example: *"Is the PD model Point-in-Time, and is this choice justified, and is it linked to expected loss requirements?"* That is really three separate retrieval targets.

When `use_query_decomposition: true` is set in `config.yaml`, the tool first asks the primary LLM to split the control into 2–3 focused sub-questions. Each sub-question is then used independently for retrieval. The results are merged and deduplicated before answer generation, giving the model richer, more complete evidence. If decomposition is disabled (the default), a single query is used.

#### Step 2 — HyDE Retrieval

Instead of searching the documents using the raw audit question (which is in English), the tool generates a **hypothetical answer** — a paragraph written in the language of your client documents (Spanish, English, etc.) that describes what a compliant policy would look like. This technique is called HyDE (Hypothetical Document Embedding).

**Why it is necessary:** the embedding model (`text-embedding-3-small`) maps text into a high-dimensional vector space. Spanish text and English text occupy slightly different regions of that space — even when they mean the same thing. If you embed the English question and search against Spanish document chunks, the L2 distances are artificially large and the wrong pages get retrieved. By generating the hypothetical in Spanish, the search becomes effectively monolingual, and the embedding distances become meaningful.

**Concrete example:**

The audit question entering the tool (always English, from the RCM CSV):
```
"We are auditing '1.1'. The requirement is: 'The bank must have a formally
approved PD model validation policy'. We must verify: 'Confirm that a written
validation policy exists, approved by the Board, and specifies validation frequency.'"
```

What HyDE produces (Spanish, mimicking the style of a real bank governance document):
```
"El banco dispone de una Política de Validación de Modelos Internos formalmente
aprobada por el Consejo de Administración en sesión ordinaria del 15 de marzo de
2024. Dicha política establece la periodicidad mínima anual para los modelos de
Probabilidad de Incumplimiento (PD), define los responsables del proceso de
validación independiente, y especifica los umbrales de aceptación para las pruebas
de backtesting, PSI y discriminación (Gini, KS). Los resultados son documentados
en informes formales elevados al Comité de Riesgos."
```

What a real chunk from your PDF looks like:
```
"La Política de Validación de Modelos, aprobada por el Directorio en febrero de
2025, establece que los modelos de PD serán sometidos a validación independiente
con periodicidad anual, cuyos resultados serán elevados al Comité de Riesgos..."
```

The HyDE paragraph and the real chunk are both in Spanish, use the same terminology (*PD*, *periodicidad anual*, *Comité de Riesgos*, *validación independiente*), and land **close together** in vector space → correctly retrieved. The original English question would land far away and likely miss this chunk entirely.

**What HyDE does NOT do:** the hypothetical paragraph is used **only as a search key**. It is never shown to the LLM that writes the final answer. The LLM only sees the real retrieved chunks from your actual documents.

**What the prompt instructs the LLM to include in the hypothetical:**
- Technical IFRS 9 vocabulary (PD, LGD, ECL, SICR...)
- Methodological synonyms (e.g. *backtesting* → also mention *PSI*, *stability*, *out-of-time validation*) — because your document may use a different term than the audit question
- Formal governance document phrasing (*aprobado por el Directorio*, *elevado al Comité*...)

This synonym expansion is critical: if the audit question says "backtesting" but your policy says "validación retrospectiva", only the hypothetical paragraph bridges that gap in the search.

#### Step 3 — Source-Balanced FAISS Retrieval

The hypothetical is embedded and used to search two separate vector stores:
- **Client index** (`faiss_index_client/`) — your PDFs, chunked and indexed per project
- **Regulations index** (`faiss_index_regs/`) — IFRS 9 and EBA guideline PDFs

Both indices are searched independently with their own caps (`client_top_k` and `regs_top_k`). This prevents the regulation documents from crowding out your client policy pages in the final context.

Chunks with an L2 distance above the threshold (default 1.8) are discarded. If none pass, the filter falls back to the top results to guarantee the model always receives something.

#### Step 4 — Multilingual Reranking

The combined pool of retrieved chunks is re-scored using a **CrossEncoder** model (`mmarco-mMiniLMv2`, multilingual). This model reads each (query, chunk) pair together and assigns a relevance score — much more accurate than embedding distance alone. The top 7 chunks survive and are passed to the answer generation step.

#### Step 5 — Answer Generation

The primary LLM (GPT-4o-mini or Gemini) receives the 7 reranked chunks as context and the audit question, and generates a structured JSON response containing:
- `classification` — DOCUMENTATION_CHECK / METHODOLOGY_CHECK / QUANTITATIVE_CHECK
- `answer` — a direct verdict with every fact cited as `[Page X of 'Filename']`
- `evidence_sources` — list of all pages used
- `compliance_verdict` — Compliant / Non-Compliant / Partial / Insufficient Info

Using JSON output eliminates the fragile text parsing that older versions used. If the model returns malformed JSON, a regex fallback activates automatically.

#### Step 6 — Expanded Retrieval on "Insufficient Info"

If the initial answer is `Insufficient Info`, the tool automatically widens the retrieval net — raising the L2 threshold by 0.4 and increasing `client_top_k` by 4 — and re-generates the answer with the larger context. This handles cases where relevant evidence exists in the documents but was filtered out by the standard threshold.

#### Step 7 — Self-Critique

The answer is scored 0–10 by a critique LLM (either the primary LLM itself or the independent judge chain, depending on the `critique_llm` setting). The score measures truthfulness and thoroughness. This is done using the `auditor_critique.j2` prompt.

#### Step 8 — Critique-Revise Loop

If the self-critique score is below `self_critique_retry_threshold` (default 6), the critique's reasoning is fed back to the primary LLM as explicit instructions to fix the specific issues. The model produces a revised answer, which is scored again. This loop repeats up to `max_revision_attempts` times. The best-scoring answer across all attempts is kept. The `Revision_Count` field in the output tells you how many cycles were triggered for each control.

#### Step 9 — Citation Verification

Every `[Page X of 'Filename']` reference in the final answer is checked against the retrieved chunks. If a citation does not appear in the actual context, it is flagged as unverified — a signal that the model may have hallucinated a page number. The `Citation_Confidence` score (0–100) reports the percentage of verified citations. Unverified citations are listed in `Unverified_Citations`.

#### Step 10 — Cross-LLM Hallucination Check

An independent judge LLM (from the Ollama → Groq → Together AI fallback chain) reads both the answer and the context and checks whether any factual claims are not supported by the retrieved documents. This produces a `hallucinated` boolean and a list of specific unsupported claims. The judge is different from the primary LLM to avoid self-evaluation bias.

#### Step 11 — Confidence Decomposition

The final `Confidence_Score` (0–100) is computed from three independent sub-scores, each measuring a different quality dimension:

| Sub-score | What it measures | Weight |
|---|---|---|
| `Retrieval_Confidence_Score` | Retrieved docs vs. target count — was there enough context? | 20% |
| `Answer_Confidence_Score` | Self-critique score × 10 — did the model reason well? | 50% |
| `Citation_Confidence` | % of cited pages verified in context — are claims grounded? | 30% |

When `enable_self_critique: false`, the Answer sub-score is absent and the remaining weights are redistributed (60% Citation, 40% Retrieval). If the cross-LLM check ran, its confidence rating is blended into the final score. A detected hallucination subtracts 30 from the cross-LLM confidence before blending.

Controls with `Confidence_Score` below `confidence_threshold` (default 60) are automatically written to `flagged_for_review.json`.

---

### Multi-project isolation

The tool supports running audits for multiple clients in one command using `run_all_projects.py`. Each project subfolder inside `documents/` is treated as a separate audit:

- Its documents are indexed into its own `faiss_index_client/` stored inside the project folder
- Its results go into its own `outputs/` subfolder
- Its document language can be set independently (e.g. BPN = Spanish, CapitalFlow = English)
- The shared regulations index is reused across all projects

Loose PDF files at the `documents/` root (not inside any subfolder) are used when running `src/run_audit.py` directly — this is the default dataset.

---

### Phase 2: Validation architecture (4 layers)

When expert reference answers are available, `validate_audit.py` adds four additional scoring layers:

| Layer | Who validates | What it checks |
|---|---|---|
| **Layer 1 — Self-Critique** | Done in Phase 1 | Truthfulness and thoroughness of the answer |
| **Layer 2 — Cross-LLM** | Done in Phase 1 | Unsupported factual claims |
| **Layer 3 — CrossEncoder** | Semantic model (no LLM) | Similarity between AI answer and expert answer |
| **Layer 4 — LLM-as-Judge** | Judge LLM | Accuracy, Stability, Drift Resistance, Guardrail tone |

The final `Weighted_Accuracy_Score` in the validation CSV weights Tier 1 controls at 3×, Tier 2 at 2×, and Tier 3 at 1×.

---

## 3. Before You Start — Know Your Resources

Answer these three questions before you choose how to run the tool.

---

**Question 1: How much time do I have?**

- **A few hours for a quick draft** → Use LITE mode in the notebook (Section 10). Skips independent checking, faster output.
- **A full day or overnight run** → Use the full pipeline (`python src/run_audit.py`). Includes all validation layers.

---

**Question 2: What hardware am I using?**

- **Standard laptop (no dedicated graphics card)** → Use Groq or Together AI as your judge model (cloud, fast, free tier). Ollama will work but will be slow.
- **Laptop with a dedicated GPU (e.g. NVIDIA)** → Ollama is a strong choice. 8B models run well, inference is fast.
- **Corporate workstation / powerful desktop** → Ollama with a larger model (llama3.1:8b or bigger) gives the best quality with no quota limits.

---

**Question 3: Does this audit need to be defensible to a regulator?**

- **Internal draft, testing, or exploration** → Strategy B (self-critique only) is acceptable. Faster, fewer API calls.
- **External submission, regulatory inspection, or model risk management report** → Strategy A (independent cross-LLM validation) is required. One AI model checking another's work is the minimum expected standard under SR 11-7 and EBA model risk guidelines.

---

**Decision summary:**

| Your situation | Recommended approach |
|---|---|
| Quick internal draft, time-pressured | Strategy B (LITE mode in notebook) |
| Full run, cloud preferred, no GPU | Strategy A + Groq or Together AI |
| Full run, data must stay local | Strategy A + Ollama |
| Regulatory submission | Strategy A, any provider |

---

## 4. Validation Strategy A — Cross-LLM as Judge

### What it means

When the primary AI model (GPT-4o-mini or Gemini) generates an audit answer, a **second, completely separate AI model** then reads both the answer and the source documents, and checks whether every factual claim in the answer is actually supported by the documents it cites.

This mimics the role of an independent reviewer in a traditional audit: someone who did not write the report reads it against the source files and flags anything that cannot be verified.

This approach is sometimes called "LLM-as-a-judge." It is the industry standard for AI-assisted audit processes that must withstand regulatory scrutiny.

The independent judge model can come from three different providers. Choose based on your setup:

---

### Option A1 — Groq (Fastest, Free, Token-Limited)

**What it is:** A cloud service that runs AI models on specialised hardware. No installation required — just an API key.

**Speed:** Very fast. Each control check takes a few seconds.

**Limitation:** The free plan allows approximately 100,000 tokens per day. A full audit of 68 controls with all validation layers uses roughly 170,000 tokens. This means you may hit the limit mid-run. The tool handles this automatically: when Groq's quota is exhausted, it switches to the next provider without stopping.

**Best for:** Pilot runs, small control sets (up to ~40 controls per day), users who can split large runs across two days.

**Setup:**
1. Go to console.groq.com and create a free account
2. Generate an API key
3. Open the `.env` file in the project folder and add: `GROQ_API_KEY=your_key_here`
4. The tool uses it automatically — no other changes needed

**Model used:** `llama-3.3-70b-versatile` (configured in `config.yaml` under `judge_llm.model`)

---

### Option A2 — Together AI (Free Tier, Automatic Fallback)

**What it is:** Another cloud API provider, also with a free tier. Used automatically as a second option if Groq's quota runs out.

**Speed:** Slightly slower than Groq; comparable answer quality.

**Limitation:** Also has daily limits. Treated by the tool as second priority in the fallback chain.

**Best for:** You do not need to actively choose this. The tool switches to it automatically when Groq is unavailable. Set it up as a backup.

**Setup:**
1. Go to api.together.xyz and create a free account
2. Generate an API key
3. Add to `.env`: `TOGETHER_API_KEY=your_key_here`

**Model used:** `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`

---

### Option A3 — Ollama (Local, Unlimited, Best for Full Runs)

**What it is:** Software that runs an open-source AI model entirely on your own machine. Nothing leaves your computer — no cloud, no quotas, no data privacy concerns.

**Speed:** Setup takes 15–30 minutes. Inference speed depends on your hardware. On a laptop without a GPU, each control check takes 15–30 seconds. On a machine with a GPU, this drops to 2–5 seconds.

**Advantage:** Once installed, you can run as many controls as you want, as many times as you want, at zero cost.

**Best for:** Large full audits (all tiers, all controls), environments where data must not leave the organisation, users where running time is not a constraint.

**Setup:**
1. Download Ollama from ollama.com and install it (standard installer, like any application)
2. Open a terminal and run: `ollama pull llama3.1:8b`
   - This downloads the model (~5 GB). Wait for it to complete.
3. Ollama runs as a background service automatically after installation
4. The tool detects it automatically — no `.env` or `config.yaml` change needed

**Choosing a model based on your hardware:**

| Your hardware | Recommended model | Command | Download size |
|---|---|---|---|
| Standard laptop, no GPU | `llama3.2` | `ollama pull llama3.2` | ~2 GB |
| Laptop with GPU (4–8 GB VRAM) | `llama3.1:8b` | `ollama pull llama3.1:8b` | ~5 GB |
| Workstation / powerful PC | `llama3.1:8b` or `mistral` | `ollama pull mistral` | ~4 GB |

To change which Ollama model the tool uses, edit `config.yaml`:
```yaml
judge_llm:
  ollama_model: "llama3.1:8b"   # change this to your downloaded model
```

---

### Provider comparison table

| Provider | Setup time | Speed | Daily token limit | Data stays local | Suitable for regulatory use |
|---|---|---|---|---|---|
| Groq | 2 minutes | Very fast | ~100k tokens | No (cloud) | Yes |
| Together AI | 2 minutes | Fast | ~100k tokens | No (cloud) | Yes |
| Ollama | 15–30 minutes | Moderate (hardware-dependent) | Unlimited | Yes | Yes |

**The tool tries providers in this order: Ollama → Groq → Together AI → fallback.**
If Ollama is running, it is used first. If it is not installed, Groq is tried. If Groq's quota is exhausted mid-run, Together AI takes over automatically. You do not need to manage this manually.

---

## 5. Validation Strategy B — Self-Critique Only

### What it means

Instead of using a second AI model to check the work, the same primary model that wrote the audit answer re-reads it alongside the source documents and scores its own response on a scale of 0 to 10 — evaluating how truthful and thorough it is.

This is faster and uses roughly three times fewer tokens than full cross-LLM validation (~800 tokens per control vs ~2,500 tokens in full mode). However, it is the same model checking its own output. This is analogous to asking an auditor to sign off on their own work without any independent review.

### When to use it

- Writing a first-pass internal draft before a full audit run
- Testing whether the retrieval system is finding the right document sections
- Exploring results quickly when time is short
- Development and debugging

### Regulatory warning

Most financial regulators and model risk frameworks (including SR 11-7, the ECB Guide on Internal Models, and EBA model risk guidelines) require **independent** validation. Self-critique alone — where the same model grades its own output — is unlikely to satisfy these requirements. **Do not submit a self-critique-only audit report to a regulator or external reviewer without also running Strategy A.**

### How to use it

In the Jupyter notebook, set the following at the top of Section 0:

```python
LITE_MODE = True          # skips cross-LLM check
SKIP_SELF_CRITIQUE = False # keeps self-critique scoring
```

In `config.yaml`, for a full pipeline run in self-critique-only mode:

```yaml
validation:
  enable_self_critique: true
  enable_cross_llm_critique: false
```

---

## 6. Step-by-Step: Running a Full Audit

### Prerequisites

Before running the tool, confirm:

- [ ] Python 3.10 or later is installed
- [ ] You have run `pip install -r requirements.txt` at least once
- [ ] Your `.env` file exists in the project root and contains at least `OPENAI_API_KEY`
- [ ] Your client PDF documents are placed in the `documents/` folder (or a subfolder for multi-project runs)
- [ ] Your audit control checklist is saved as `inputs/rcm_input.csv`
- [ ] (Optional) Expert reference answers are in `inputs/rcm_expert_answer.csv`

---

### Path 1: Single project — audit only

The most common approach for a single client. Skips validation (no expert answers needed).

```bash
python src/run_audit.py --no-validation
```

Documents are read from the folder specified by `paths.documents_folder` in `config.yaml` (default: `documents/`). Only PDF files at the root of that folder are loaded — subfolders are not scanned.

---

### Path 2: Full pipeline — audit and validate in one command

Use this when you have expert reference answers and want the full validation report.

```bash
python src/run_audit.py
```

This single command:
1. Reads all documents and builds the search index
2. Processes every control in `rcm_input.csv`
3. Generates audit answers with evidence citations
4. Runs self-correction if answers score below threshold
5. Runs the independent judge model for hallucination checks
6. Saves all results into a timestamped folder under `outputs/`
7. Automatically runs the validation step against expert answers

---

### Path 3: Resume an interrupted run

If a run is interrupted (network error, power failure, etc.), every completed row is saved to a `checkpoint.json` file inside the run folder. Resume from where it left off:

```bash
python src/run_audit.py --resume outputs/20260320T143022Z/ --no-validation
```

The tool skips already-completed rows, continues from the next one, and saves the final `audit_results.json` when done. The checkpoint file is deleted on successful completion.

---

### Path 4: Multiple projects in one command

When you have multiple client folders inside `documents/`, run all of them sequentially with per-project document isolation:

```bash
python run_all_projects.py
```

This script:
- Detects each subfolder in `documents/` (e.g. `BPN/`, `CapitalFlow/`, `UCI/`)
- Builds a separate FAISS index inside each project folder
- Runs `src/run_audit.py --no-validation` for each project
- Moves the results into `documents/PROJECT/outputs/TIMESTAMP/`

Per-project settings (such as document language) are configured in the `PROJECT_OVERRIDES` dictionary at the top of `run_all_projects.py`:

```python
PROJECT_OVERRIDES = {
    "BPN":         {"rag_settings": {"document_language": "Spanish"}},
    "CapitalFlow": {"rag_settings": {"document_language": "English"}},
    "UCI":         {"rag_settings": {"document_language": "Spanish"}},
}
```

---

### Path 5: Validation only

If you already have audit results and want to run validation separately:

```bash
python src/validate_audit.py
```

This auto-detects the latest run folder in `outputs/`.

---

### Path 6: Interactive notebook for spot checks

For checking individual controls before committing to a full run:

```bash
jupyter notebook notebooks/interactive_audit.ipynb
```

See Section 10 of this manual for how to use the notebook.

---

### Filtering by tier

Edit `config.yaml` before running:

```yaml
filtering:
  tier: '1'    # options: 'all', '1', '2', '3'
```

Tier 1 is the default. Changing to `'all'` processes all 67 controls; expect roughly 3× the time and tokens.

---

### What to expect during a run

```
Run ID: 20260320T143022Z
Output folder: outputs/20260320T143022Z/
Initializing RAG Engine...
Filtered for Tier: 1. Remaining rows: 32
Processing 32 rows...
Processing row 1/32 (CSV row 2)...
  [Decomposition] 3 sub-queries for 6.2
Processing row 2/32 (CSV row 3)...
  [Remediation] Score 4/10 < 6 — revising answer for 7.3 (attempt 1/2)...
  [Remediation] Best score after 1 revision(s): 7/10 for 7.3
  [Citation] 1 unverified citation(s) for 9.3: ['Page 12 of Marco Metodologico.pdf']
...
Audit results saved to outputs/20260320T143022Z/audit_results.json
Flagged 5/32 controls for human review
```

A full run of 32 Tier 1 controls typically takes:
- **With Groq as judge, no self-correction triggered:** 15–25 minutes
- **With self-correction on ~30% of rows:** add 5–10 minutes
- **With Ollama on CPU:** 60–120 minutes
- **With Ollama on GPU:** 20–40 minutes

---

## 7. Understanding Your Results

### Compliance Verdict

Each control receives one of four verdicts:

| Verdict | Meaning |
|---|---|
| **Compliant** | The documents clearly evidence that the requirement is met |
| **Non-Compliant** | The documents show the requirement is not met, or evidence contradicts it |
| **Partial** | Some aspects are evidenced but others are missing or unclear |
| **Insufficient Info** | The documents do not contain enough information to make a determination |

An "Insufficient Info" verdict does not mean the bank is non-compliant — it means the documents provided to the tool do not allow a conclusion to be drawn. Additional documents may be needed. Note that the tool automatically attempts one expanded retrieval before settling on this verdict.

---

### Check Classification

Each control is automatically classified into one of three categories, which determines how strictly the AI scrutinises the evidence:

| Classification | What it checks | Example |
|---|---|---|
| **DOCUMENTATION_CHECK** | Does a policy or document exist? Was it approved? | "Is there a formal ECL policy approved by the Board?" |
| **METHODOLOGY_CHECK** | Is the approach described, justified, and logical? | "Is the SICR definition documented with clear criteria?" |
| **QUANTITATIVE_CHECK** | Are specific numbers validated and backtested? | "Is the PD threshold for SICR supported by data?" |

QUANTITATIVE_CHECKs apply the strictest scrutiny. The AI requires numerical parameters, empirical validation results, and confidence intervals to be cited explicitly.

---

### Confidence Score (0–100) and Sub-Scores

The overall `Confidence_Score` is a weighted composite of three independent sub-scores:

| Sub-score field | What it measures | Ideal value |
|---|---|---|
| `Retrieval_Confidence_Score` | Ratio of retrieved chunks to the target count — was there enough context? | 100 (retrieved ≥ rerank_top_k) |
| `Answer_Confidence_Score` | Self-critique score × 10 — did the model reason correctly and thoroughly? | 80–100 |
| `Citation_Confidence` | % of `[Page X of 'File']` citations verified against retrieved context | 100 (all grounded) |

- **80–100 overall:** High confidence. Well-supported answer, no unverified citations.
- **60–79:** Moderate confidence. Review the answer but it is likely reliable.
- **Below 60:** Low confidence. Automatically flagged for human review.

A low `Retrieval_Confidence_Score` with a high `Answer_Confidence_Score` suggests the model answered well but from thin evidence — verify that the right documents are indexed. A low `Citation_Confidence` with a reasonable overall score is a hallucination warning — inspect `Unverified_Citations`.

---

### Self-Correction Indicators

| Field | What it tells you |
|---|---|
| `Revision_Count` | 0 = answer was accepted on first try; 1 or 2 = revision cycles were triggered |
| `Retrieval_Expanded` | true = the "Insufficient Info" expanded retrieval fallback was used |
| `Unverified_Citations` | Page references in the answer not found in the retrieved context |

A `Revision_Count` of 2 on many controls may indicate that the document language setting is wrong, or that the documents don't contain sufficient evidence for those controls.

---

### Validation Scores (val_metrics CSV)

When expert reference answers are available, the validation report adds four additional dimensions:

| Score | What it measures | Scale |
|---|---|---|
| **Cross-Encoder Score** | Semantic similarity between AI answer and expert answer | 0–100 |
| **Validation Score (LLM-as-judge)** | How accurately the AI captured the expert's factual content | 0–100 |
| **Stability Score** | Whether the AI's reasoning is logically consistent | 0–100 |
| **Drift Resistance Score** | Whether the AI stayed grounded in the documents (didn't hallucinate) | 0–100 |
| **Guardrail Score** | Whether the AI maintained professional, objective tone | 0–100 |

The overall **Weighted Accuracy Score** multiplies each control's accuracy score by its tier weight (Tier 1 = 3×, Tier 2 = 2×, Tier 3 = 1×) and averages the result.

---

### Flagged Controls

Any control that meets one or more of these conditions is written to `flagged_for_review.json`:

- Self-critique score below 6 out of 10
- Confidence score below 60 out of 100
- Verdict of "Insufficient Info"
- Independent checker detected a claim not supported by the documents

Flagged controls are not failures — they are the tool's way of saying "a human auditor should look at this one." In a well-functioning audit, expect 10–30% of controls to be flagged.

---

## 8. The Output Folder — What Each File Means

Each run creates its own folder inside `outputs/`, named by the timestamp when the run started. For multi-project runs, this folder is moved inside the project's own directory.

```
outputs/20260320T143022Z/          ← single project run
documents/BPN/outputs/20260320T143022Z/    ← multi-project run
  audit_results.json
  run_manifest.json
  flagged_for_review.json
  val_metrics_20260320T143022Z_ollama_llama3.1-8b.csv
  checkpoint.json                  ← only present during an active or interrupted run
```

---

### audit_results.json

The main output. Contains one entry per control with all fields described in Section 7. Convert to Excel using:

```bash
python json_to_xlsx.py   # edit the FILES list inside the script first
```

---

### checkpoint.json

Written after every completed row during a run. If the run completes successfully, this file is deleted. If the run is interrupted, it remains and can be used to resume:

```bash
python src/run_audit.py --resume outputs/20260320T143022Z/ --no-validation
```

Do not delete this file manually while a run is in progress or if you intend to resume.

---

### run_manifest.json

A record of exactly how this run was produced. Contains:

- The run ID and UTC timestamp
- Which AI model was used as primary and as judge
- Which tier was audited
- SHA-256 checksums (fingerprints) of every document that was read

The document fingerprints are important for regulatory defensibility: they prove that the documents used in the audit have not been modified after the fact. If a regulator asks "which version of the policy was audited?", the SHA-256 hash in the manifest answers that question definitively.

---

### flagged_for_review.json

A filtered list of only the controls that require human attention, with the specific reason each was flagged. Use this as your starting point for manual review — focus on flagged controls first.

---

### val_metrics_*.csv

The validation report. The filename includes the UTC timestamp and the name of the judge model used (e.g. `ollama_llama3.1-8b`), so you can always trace a report back to the exact model that produced it.

Key columns:
- `Control Reference` — the control ID
- `Compliance_Verdict` — Compliant / Non-Compliant / Partial / Insufficient Info
- `Validation_Score_LLM_as_judge` — overall accuracy score vs expert (0–100)
- `Cross_Encoder_Score` — semantic similarity vs expert (0–100)
- `Reasoning_LLM_as_judge` — the judge model's written explanation of the score
- `Answers based on Clients data` — the expert reference answer (for comparison)

---

## 9. Configuring the Tool

All settings are in `config.yaml` in the project root. Open it with any text editor.

### Which controls to run

```yaml
filtering:
  tier: '1'    # change to '2', '3', or 'all'
```

### Document language (per project)

```yaml
rag_settings:
  document_language: "Spanish"   # or "English", "French", "Portuguese", etc.
```

This controls the language of the HyDE hypothetical used for retrieval. It must match the language of your client documents. For multi-project runs, override this per project in `run_all_projects.py` instead of changing `config.yaml`.

### Enable query decomposition

For complex compound controls, query decomposition retrieves richer evidence by splitting the control into sub-questions:

```yaml
rag_settings:
  use_query_decomposition: false   # set true to enable; adds ~1 LLM call per row
```

### Enable or disable self-correction

```yaml
validation:
  enable_self_critique: true
  self_critique_retry_threshold: 6   # revise if score < this (0-10)
  max_revision_attempts: 2           # max revision cycles before accepting best
  critique_llm: "primary"            # "primary" = same model; "judge" = independent chain
```

Setting `critique_llm: "primary"` (default) uses GPT-4o-mini to score its own answers — faster and simpler. Setting `"judge"` uses the Ollama/Groq chain for a more independent evaluation.

### Enable or disable cross-LLM hallucination check

```yaml
validation:
  enable_cross_llm_critique: true    # requires a judge LLM provider to be available
  confidence_threshold: 60.0         # controls below this are flagged for human review
```

### Skip or include validation step

```yaml
audit_trail:
  run_validation: false   # set true to run validate_audit.py after the audit
```

This can also be overridden at runtime:
```bash
python src/run_audit.py --no-validation
```

### Turn off regulation documents

```yaml
rag_settings:
  use_regulations: false   # saves ~1,500 tokens per control; faster runs
```

### Changing the Ollama model

```yaml
judge_llm:
  ollama_model: "llama3.1:8b"   # must match a model pulled with 'ollama pull'
```

---

## 10. Using the Interactive Notebook for Spot Checks

The notebook (`notebooks/interactive_audit.ipynb`) lets you test individual controls without running the full pipeline.

### Starting the notebook

```bash
jupyter notebook notebooks/interactive_audit.ipynb
```

### Always run Section 0 first

Section 0 loads the environment, connects to the AI models, and builds the document search index. Every other section depends on it. Run it once at the start of each session.

### Section guide

| Section | What it does | When to use it |
|---|---|---|
| **0. Setup** | Loads everything; set LITE_MODE here | Always first |
| **1. RAG Debug** | Shows which document chunks are retrieved for a query | If answers seem off — check retrieval first |
| **2. Single Control (LITE)** | Runs one control, answer only | Quick check; ~800 tokens |
| **3. Single Control (FULL)** | Runs one control with all validation layers | Before full run; ~2,500 tokens |
| **3b. Custom Question** | Test any question without needing the CSV | Ad-hoc exploration |
| **4. Compare vs Expert** | Scores AI answer against expert reference | Quality check on a single control |
| **5. Adversarial Test** | Tests edge cases | QA and robustness checking |
| **6. Batch (N rows)** | Runs N controls and saves results | Mini audit run |
| **7. Full Validation** | Runs validate_audit on the latest results | After Section 6 |
| **8. Ollama Test** | Checks if Ollama is running and the model works | First-time Ollama setup verification |

### LITE mode vs FULL mode

```python
LITE_MODE = True    # True = answer only (~800 tokens/row)
                    # False = answer + self-critique + cross-LLM (~2,500 tokens/row)
```

---

## 11. Troubleshooting

### "Connection error" in AI_Answer column

The primary AI model (OpenAI) could not be reached. Check:
1. Is `OPENAI_API_KEY` in your `.env` file?
2. Is the `.env` file in the project root folder (same level as `config.yaml`)?
3. Do you have internet access?

### Ollama not detected / falls back to Groq

The tool only uses Ollama if it is running. Check:
```bash
ollama list    # should show your downloaded models
```
If Ollama is not running, start it:
```bash
ollama serve
```

### Many controls returning Revision_Count = 2

This means most answers scored below 6/10 on self-critique and used both revision attempts. Likely causes:
- Wrong `document_language` (retrieval is finding irrelevant chunks)
- Documents do not contain sufficient evidence for these controls (correct verdict is "Insufficient Info")
- `enable_self_critique: true` with a very strict judge — try lowering `self_critique_retry_threshold` to 5

### High Unverified_Citations count

The model is citing pages that were not in the retrieved context. Possible causes:
- The LLM is hallucinating page numbers (flag for human review)
- The documents use non-standard page numbering (e.g. Roman numerals, section numbers) that don't match PDF page indices

### Run is very slow

If Ollama is being used as judge on a CPU-only machine, each control check takes 15–30 seconds. Options:
- Use Groq instead (add `GROQ_API_KEY` to `.env`)
- Disable self-critique: `enable_self_critique: false` in `config.yaml`
- Use a smaller Ollama model: `ollama pull llama3.2` (2 GB, faster)

### Groq quota exhausted mid-run

You will see `Provider exhausted. Trying next...` — the tool automatically switches to Together AI and continues. If Together AI also fails, add Ollama as a fallback (see Section 4, Option A3).

### Run interrupted — how to resume

Find the run folder (it will contain a `checkpoint.json` file):
```bash
python src/run_audit.py --resume outputs/20260320T143022Z/ --no-validation
```

For a multi-project run, the output folder is inside the project folder:
```bash
python src/run_audit.py --resume documents/BPN/outputs/20260320T143022Z/ --no-validation
```

### Results look irrelevant — wrong document sections retrieved

Open the notebook and run Section 1 (RAG Debug) with your control's question. If the retrieved chunks are wrong, try:
- Checking `document_language` matches your documents
- Enabling `use_query_decomposition: true` for compound controls
- Increasing `client_top_k` to retrieve more candidates before reranking

### FAISS index seems stale after adding new documents

Delete the index folder for that project and let the tool rebuild it:
```bash
# For a single project run (default index)
rmdir /s /q faiss_index_client

# For a specific project
rmdir /s /q "documents/BPN/faiss_index_client"
```
The next run will rebuild the index from scratch.

---

## 12. Quick Reference Card

### Commands

| Task | Command |
|---|---|
| Audit only (no validation) | `python src/run_audit.py --no-validation` |
| Full audit + validation | `python src/run_audit.py` |
| Resume interrupted run | `python src/run_audit.py --resume outputs/TIMESTAMP/ --no-validation` |
| All projects in documents/ | `python run_all_projects.py` |
| Validation only (latest run) | `python src/validate_audit.py` |
| Convert results to Excel | `python json_to_xlsx.py` (edit FILES list inside) |
| Client policy summary | `python src/run_summary.py` |
| Interactive notebook | `jupyter notebook notebooks/interactive_audit.ipynb` |

### Key files

| File | What you edit / read |
|---|---|
| `config.yaml` | All settings: tier, language, models, thresholds |
| `.env` | API keys |
| `inputs/rcm_input.csv` | Your audit control checklist |
| `inputs/rcm_expert_answer.csv` | Expert reference answers (for validation) |
| `documents/` | Loose PDFs for single-project runs |
| `documents/PROJECT/` | PDFs for multi-project runs |
| `outputs/<timestamp>/` | All results from a single-project run |
| `documents/PROJECT/outputs/<timestamp>/` | All results from a multi-project run |
| `run_all_projects.py` | Per-project language overrides |

### config.yaml most-used settings

| Setting | Default | What it does |
|---|---|---|
| `filtering.tier` | `'1'` | Which tier to audit (`'all'`, `'1'`, `'2'`, `'3'`) |
| `rag_settings.document_language` | `Spanish` | Language of your client documents |
| `rag_settings.use_regulations` | `true` | Include EBA regulation docs in search |
| `rag_settings.use_query_decomposition` | `false` | Split compound controls into sub-queries |
| `validation.enable_self_critique` | `false` | Score each answer 0–10 |
| `validation.critique_llm` | `primary` | Which LLM scores the answer (`primary` or `judge`) |
| `validation.self_critique_retry_threshold` | `6` | Trigger revision if score below this |
| `validation.max_revision_attempts` | `2` | Max revision cycles per control |
| `validation.enable_cross_llm_critique` | `false` | Independent hallucination check |
| `validation.confidence_threshold` | `60.0` | Below this → flagged for human review |
| `audit_trail.run_validation` | `false` | Run validate_audit.py after audit |
| `judge_llm.ollama_model` | `llama3.1:8b` | Which local Ollama model to use |

### audit_results.json — key output fields per control

| Field | Description |
|---|---|
| `Classification` | DOCUMENTATION_CHECK / METHODOLOGY_CHECK / QUANTITATIVE_CHECK |
| `AI_Answer` | Full written answer with page citations |
| `Compliance_Verdict` | Compliant / Non-Compliant / Partial / Insufficient Info |
| `Validation_Score` | Self-critique score 0–10 |
| `Confidence_Score` | Weighted composite 0–100 |
| `Retrieval_Confidence_Score` | Context coverage sub-score 0–100 |
| `Answer_Confidence_Score` | Reasoning quality sub-score 0–100 |
| `Citation_Confidence` | % of citations grounded in retrieved context |
| `Unverified_Citations` | Page references not found in context |
| `Revision_Count` | Number of self-correction cycles triggered |
| `Retrieval_Expanded` | Whether expanded retrieval fallback was used |
| `Cross_LLM_Hallucinated` | True if independent checker flagged unsupported claims |
| `Cross_LLM_Concerns` | List of specific unsupported claims found |

### Judge LLM fallback order

1. **Ollama** (local) — if installed and running
2. **Groq** — if `GROQ_API_KEY` is in `.env`
3. **Together AI** — if `TOGETHER_API_KEY` is in `.env`
4. **Secondary provider** (Google or OpenAI, whichever is not primary)
5. **Primary LLM** — last resort

---

## Annex A — Architecture Diagram

The diagram below shows the complete data flow of the tool, from entry points through the RAG engine, RCM engine, validation layers, and optional output modules.

![Architecture Diagram](images/mermaid-diagram-2026-03-27-154310.png)

> To regenerate the diagram: copy the contents of `marmaid_code_arquitecture.txt` into [mermaid.live](https://mermaid.live) and export as PNG.

---

## Annex B — Technical Module Reference

This annex describes what each Python source file does at the code level. It is intended for developers, validators, and anyone who needs to understand, audit, or modify the pipeline.

---

### B.1 `config.py` — Configuration Loader

**What it does:** Loads `config.yaml` and `.env` into a global `CONFIG` dict accessible by all other modules. Resolves all relative paths to absolute paths anchored at the project root.

**Key behaviour:**
- The environment variable `AUDIT_CONFIG_PATH` allows a subprocess (e.g. `run_all_projects.py`) to point the tool at a different YAML file without modifying the original `config.yaml`. This is how per-project configurations are injected.
- `PROJECT_ROOT` is computed as the parent of the `src/` directory and used throughout the codebase to build absolute paths.
- All path values in `config.yaml` (documents folder, FAISS index paths, CSV paths) are resolved relative to `PROJECT_ROOT` at load time, so the tool can be invoked from any working directory.

---

### B.2 `llm_factory.py` — LLM and Embeddings Factory

**What it does:** Provides three public functions — `get_llm()`, `get_embeddings()`, and `get_judge_llm()` — that instantiate the correct AI provider based on `config.yaml`.

**Primary LLM (`get_llm`):**
- Returns a `ChatOpenAI` or `ChatGoogleGenerativeAI` instance depending on `llm_settings.provider`.
- Temperature is always `0.0` to ensure deterministic, reproducible audit answers.

**Embeddings (`get_embeddings`):**
- Returns `OpenAIEmbeddings` (`text-embedding-3-small`) or `GoogleGenerativeAIEmbeddings` (`embedding-001`).
- Used by the RAG engine to embed document chunks and queries into the same vector space.

**Judge LLM (`get_judge_llm`) — Fallback Chain:**

This is the most complex part of the factory. The judge LLM is intentionally different from the primary LLM so it can provide independent validation. The resolution order is controlled by `judge_llm.provider` in `config.yaml`:

| Step | Provider | How it is tried | Why |
|---|---|---|---|
| 1 | **Ollama** | Sends a `"ping"` message to `localhost:11434` | Local, zero cost, zero quota, fully private |
| 2 | **Groq** | Checks for `GROQ_API_KEY` in `.env` | Free cloud, fast, 100k tokens/day |
| 3 | **Together AI** | Checks for `TOGETHER_API_KEY` in `.env` | Free model available (`Llama-3.3-70B-Instruct-Turbo-Free`) |
| 4 | **Secondary provider** | Google if primary is OpenAI, or vice versa | Still independent from primary |
| 5 | **Primary LLM** | Always available | Least independent, last resort |

The `_try_*` private functions each return `None` on failure rather than raising, so the chain advances silently. The `get_fallback_judge_llm(skip=[...])` function is used mid-run when a provider exhausts its quota (daily token limit, credit limit, or auth failure) — it rebuilds the chain excluding the failed provider so the run continues without interruption.

The label of whichever provider was selected is stored in `_judge_llm_label` and used to name the output CSV (`val_metrics_<timestamp>_<judge>.csv`), making it auditable which model performed the validation.

---

### B.3 `rag_engine.py` — Retrieval-Augmented Generation Engine

**What it does:** Builds and manages the FAISS vector stores, and implements the full retrieval pipeline: document loading → chunking → embedding → indexing → HyDE query generation → similarity search → L2 threshold filtering → CrossEncoder reranking.

#### Document Loading and Indexing

`load_documents_from_folder(folder_path)` iterates over all PDFs in a folder using `PyPDFLoader` (from LangChain). Each page becomes a separate `Document` object with `metadata['page']` and `metadata['source']` set.

`_build_or_load_index(index_name, folder_path)` either loads an existing FAISS index from disk (`FAISS.load_local`) or builds a new one from scratch. When building:
1. Documents are split into overlapping chunks (`RecursiveCharacterTextSplitter`, default 1200 chars / 300 overlap).
2. Chunks are embedded in batches of 10, with a 5-second delay between batches to avoid API rate limits.
3. The index is saved to disk so subsequent runs skip re-embedding.

Two indices are maintained separately:
- `faiss_index_client/` — per-project client documents (path is configurable to enable isolation)
- `faiss_index_regs/` — shared regulation documents (IFRS 9 / EBA, path is fixed)

#### HyDE Query Generation (`generate_search_query`)

Instead of embedding the raw English audit question and searching Spanish documents, the engine first asks the primary LLM to write a **hypothetical answer** in the document language (`document_language` from config). This hypothetical paragraph uses the same vocabulary, phrasing, and technical terms that the real bank policy document would use — dramatically improving retrieval accuracy for cross-lingual setups.

Example: for the audit question *"Does the bank document its PD model validation frequency?"*, HyDE produces a Spanish paragraph like: *"El banco lleva a cabo un proceso de validación anual del modelo de PD en el que se evalúa la estabilidad del ranking..."* — which is much closer in embedding space to the actual policy document text than the original English question.

#### Source-Balanced Retrieval

Both the client and regulations indices are queried **independently** using the HyDE query:
- Client: up to `client_top_k` (default 6) nearest neighbours
- Regulations: up to `regs_top_k` (default 2) nearest neighbours

The independent caps prevent regulations from crowding out the client policy content. If regulations were queried together with client docs in a single pool, the IFRS 9 standard text (which is highly uniform and always relevant) would often rank above the specific client policy pages that actually answer the audit question.

#### L2 Threshold Filtering (`_filter_by_threshold`)

After retrieval, chunks with L2 distance > `retrieval_score_threshold` (default 1.8) are discarded. The threshold is permissive relative to standard monolingual setups because cross-lingual embeddings — even with HyDE — naturally produce higher L2 distances than same-language pairs. A value of 1.8 corresponds roughly to cosine similarity of –0.62, meaning only true noise is discarded. If all chunks fail the threshold (which can happen for very specific controls), the filter is bypassed and the closest results are used to ensure the LLM always receives some context.

#### CrossEncoder Reranking

After threshold filtering, the combined pool (client + regulation chunks) is reranked using `sentence_transformers.CrossEncoder` with the model `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. This is a multilingual MS MARCO-trained model that scores each (original query, chunk) pair jointly — capturing semantic relevance that bi-encoder FAISS cannot.

The top `rerank_top_k` (default 7) chunks after reranking are passed to the LLM. The reranker is lazy-loaded (on first use) and cached for the remainder of the run.

---

### B.4 `rcm_engine.py` — Core Audit Logic

**What it does:** Implements the `RcmAuditor` class whose `process_row(row)` method takes a single RCM CSV row and returns a result dict with the AI answer, verdict, and all quality scores. This is the heart of the pipeline.

#### Prompt Injection Protection

Before any chunk is passed to the LLM, `_sanitize_chunk(text)` scans it against a list of regular expressions that match common prompt injection patterns (`IGNORE PREVIOUS INSTRUCTIONS`, `jailbreak`, `DAN mode`, etc.). Matches are replaced with `[REDACTED]`. This prevents a malicious actor from embedding instructions inside a PDF that would hijack the LLM's behaviour.

#### `_build_context(retrieved_docs)` → `(context_text, evidence_dict)`

Formats the retrieved chunks into a numbered context string, with each chunk prefixed by its page marker `[Page N of 'Filename']`. Also builds an `evidence_dict` mapping filenames to sets of page numbers — used later to populate the `Evidence_Sources` field.

#### `_parse_response(full_response)` → `(answer, evidence, verdict, classification)`

The primary LLM is instructed to return **only valid JSON** via `auditor_response.j2`. This function parses that JSON and extracts the four structured fields. If JSON parsing fails (e.g. the model wraps the response in a markdown code block, or produces malformed JSON), it falls back automatically to the original regex-based extraction — searching for `<answer>`, `<evidence_sources>`, and `**COMPLIANCE VERDICT:**` tags. This dual approach means a badly formatted model response never silently loses the answer.

#### `_verify_citations(answer, context_text)` → `{total, verified, unverified, confidence}`

Uses a regular expression to find every `[Page X of 'Filename']` reference in the answer, then checks whether the exact same string appears in `context_text`. A citation is **verified** if the page was actually retrieved (i.e. it exists in the context); **unverified** citations are page numbers the model may have hallucinated. Returns a confidence percentage: `verified / total * 100`. Controls with unverified citations are logged and their `Citation_Confidence` sub-score is reduced.

#### `_run_self_critique(context_text, query, answer)` → `{score, reasoning, hallucination_rate}`

Renders `auditor_critique.j2` with the context and answer, then invokes the critique LLM (either the primary LLM or the judge chain, depending on `critique_llm` config). The template asks the LLM to score the answer 0–10 on truthfulness and completeness, explain its reasoning, and estimate a hallucination rate (0.0–1.0). Returns `score: 0` on any failure so the pipeline continues gracefully.

#### `_cross_llm_critique(context_text, answer)` → `{hallucinated, unsupported_claims, confidence_in_answer}`

Uses a separate LLM (from the judge chain) to independently check whether the answer contains claims not supported by the retrieved context. The prompt explicitly instructs the checker *not* to penalise for missing information — only to flag invented facts. Returns a boolean `hallucinated` flag and a list of specific unsupported claims. Uses a persistent provider queue across rows: if Groq exhausts its daily quota on row 30, the queue advances to Together AI for all remaining rows without interruption.

#### `_decompose_query(design_assessment, test_procedure)` → `[sub_query, ...]` or `None`

When `use_query_decomposition: true`, asks the primary LLM to split the compound audit control into 2–3 focused, independent sub-questions, returned as a JSON array. Only enabled explicitly because it adds one extra LLM call per row. If decomposition fails (malformed JSON, API error), returns `None` and the caller falls back to single-query retrieval.

#### `process_row(row)` — The 8-Step Pipeline

```
Step 1 — Initial retrieval
  ├─ Optionally decompose into sub-queries (_decompose_query)
  ├─ Retrieve per sub-query and merge, OR single retrieve
  └─ Build context string → generate answer via auditor_response.j2

Step 2 — Expanded retrieval (only if verdict == "Insufficient Info")
  ├─ Widen search: threshold +0.4, client_top_k +4
  ├─ Re-retrieve with looser filter
  └─ Regenerate answer with wider context

Step 3 — Self-critique (if enable_self_critique: true)
  └─ Score the answer 0–10 via auditor_critique.j2

Step 4 — Critique-Revise loop (if score < self_critique_retry_threshold)
  ├─ Render auditor_revision.j2 with previous answer + critique reasoning
  ├─ Re-score the revised answer
  ├─ Keep the best (highest-scored) answer across all attempts
  └─ Repeat up to max_revision_attempts times

Step 5 — Citation verification
  └─ Check every [Page X of 'File'] against context_text

Step 6 — Cross-LLM hallucination check (if enable_cross_llm_critique: true)
  └─ Independent judge LLM checks for unsupported claims

Step 7 — Confidence decomposition
  ├─ Retrieval_Confidence  = min(num_retrieved / rerank_top_k, 1.0) × 100   [weight 20%]
  ├─ Answer_Confidence     = (critique_score / 10) × 100                    [weight 50%]
  ├─ Citation_Confidence   = verified_citations / total_citations × 100      [weight 30%]
  ├─ base_confidence       = 0.5 × Answer + 0.3 × Citation + 0.2 × Retrieval
  └─ If cross-LLM ran: final = (base_confidence + cross_conf) / 2
     (cross_conf is penalised –30 if hallucinated flag is true)

Step 8 — Build result dict
  └─ All input row fields + all computed fields returned
```

If `enable_self_critique` is `false`, the Answer sub-score is absent and the weight redistribution is: 60% Citation + 40% Retrieval.

---

### B.5 `run_audit.py` — Entry Point

**What it does:** Orchestrates a full audit run — reads the RCM CSV, loops over rows, calls `RcmAuditor.process_row()` for each, and saves all output files into a timestamped folder.

**Key mechanics:**

- **Timestamped run folder:** `outputs/YYYYMMDDTHHMMSSZ/`. Created fresh for each new run. All output files go here.
- **Tier filtering:** Reads `filtering.tier` from config. If not `'all'`, filters the DataFrame to rows matching the tier value before processing.
- **Checkpoint system:** After each row, `results` (a list of dicts) is serialised to `checkpoint.json` in the run folder. If the run is interrupted, it can be resumed with `--resume <run_folder>`. On resume, `done_refs` is built from the checkpoint and already-completed rows are skipped. The checkpoint is deleted on successful completion.
- **SHA-256 document hashing:** Before processing, every file in `documents/` and `regulations/` is hashed. Hashes are saved in `run_manifest.json`. This allows proving to regulators that the same documents were used for the audit as were present at run time.
- **Flagging logic:** After the main loop, each result is checked against `flag_score_threshold` (self-critique score), `confidence_threshold`, `Insufficient Info` verdict, and `Cross_LLM_Hallucinated`. Any control meeting one or more conditions is written to `flagged_for_review.json`.
- **Validation trigger:** If `audit_trail.run_validation: true` (or `--no-validation` was not passed), `validate_audit()` is called automatically after the audit loop. The output CSV is written into the same run folder.

---

### B.6 `run_all_projects.py` — Multi-Project Orchestrator

**What it does:** Discovers project subfolders inside `documents/`, writes a temporary per-project `config.yaml` for each, and runs `src/run_audit.py --no-validation` as a subprocess for each project. After the run, it moves the timestamped output folder from `outputs/` into `documents/<PROJECT>/outputs/`.

**Per-project isolation:**
- The temporary config sets `paths.documents_folder` to the project subfolder and `paths.faiss_index_client` to `documents/<PROJECT>/faiss_index_client/`.
- The subprocess receives `AUDIT_CONFIG_PATH=<temp_yaml_path>` as an environment variable, which `config.py` reads at import time.
- Because `faiss_index_client` is per-project, embedding builds for one project never interfere with another.

**Language overrides:**
- The `PROJECT_OVERRIDES` dict at the top of the script lets you set per-project `document_language` (and any other config key). For example, CapitalFlow uses `"English"` while BPN and UCI use `"Spanish"`.
- These overrides are merged into the temporary config before it is written to disk.

---

### B.7 `validate_audit.py` — Validation Phase

**What it does:** Compares the AI-generated answers from `audit_results.json` against expert reference answers from `rcm_expert_answer.csv`. Produces `val_metrics_<timestamp>_<judge>.csv` with per-control scores on four dimensions.

**4-Layer validation:**

| Layer | Implemented by | When it runs |
|---|---|---|
| Self-critique | `rcm_engine.py` / `_run_self_critique` | During Phase 1 (per row) |
| Cross-LLM hallucination check | `rcm_engine.py` / `_cross_llm_critique` | During Phase 1 (per row) |
| CrossEncoder semantic similarity | `sentence_transformers.CrossEncoder` | During Phase 2 (no LLM cost) |
| LLM-as-judge vs expert answer | Judge LLM via `auditor_critique.j2` logic | During Phase 2 |

The CrossEncoder layer computes a cosine similarity between the AI answer embedding and the expert answer embedding, producing `Cross_Encoder_Score` (0–100). This layer costs no LLM tokens.

The LLM-as-judge layer asks the judge LLM to score the AI answer on four dimensions relative to the expert answer:
- **Accuracy** — Does the AI answer reach the same conclusions as the expert?
- **Stability** — Is the reasoning internally consistent?
- **Drift Resistance** — Does the answer stay grounded in the documents (no hallucination drift)?
- **Guardrail Effectiveness** — Is the tone professional and suitable for a regulatory submission?

The final `Weighted_Accuracy_Score` applies tier weights (Tier 1 × 3, Tier 2 × 2, Tier 3 × 1) to produce a risk-adjusted overall metric.

---

### B.8 Prompt Templates (`templates/`)

| File | Used by | Purpose |
|---|---|---|
| `auditor_response.j2` | `rcm_engine.py` / `process_row` | Main audit answer generation. Receives `context`, `design_assessment`, `test_procedure`. Instructs the LLM to return a JSON object with `classification`, `answer`, `evidence_sources`, `compliance_verdict`. |
| `auditor_critique.j2` | `rcm_engine.py` / `_run_self_critique` | Self-critique scoring. Receives `context`, `query`, `answer`. Instructs the LLM to return JSON with `score` (0–10), `reasoning`, `hallucination_rate` (0.0–1.0). |
| `auditor_revision.j2` | `rcm_engine.py` / critique-revise loop | Revision after a low critique score. Receives `context`, `design_assessment`, `test_procedure`, `previous_answer`, `critique_reasoning`. Returns same JSON schema as `auditor_response.j2`. |
| `client_summary.j2` | `run_summary.py` | Generates a markdown summary of the client's policies across 13 IFRS 9 governance topics. Receives the full retrieved context. |

All answer and revision templates use **structured JSON output** (`{"classification": ..., "answer": ..., "evidence_sources": ..., "compliance_verdict": ...}`). This replaces the original XML-tag + regex approach and is significantly more robust to model variability. The `_parse_response()` function in `rcm_engine.py` retains the regex fallback for backward compatibility.

---

### B.9 Data Flow Summary

```
rcm_input.csv
    │
    ▼  (per row)
run_audit.py ──► RcmAuditor.process_row()
                     │
                     ├─ 1. (optional) _decompose_query()   ← primary LLM, 1 extra call
                     ├─ 2. RagEngine.retrieve()
                     │       ├─ generate_search_query()    ← HyDE: primary LLM
                     │       ├─ FAISS search (client + regs separately)
                     │       ├─ _filter_by_threshold()     ← L2 ≤ 1.8
                     │       └─ CrossEncoder rerank         ← mmarco model, no LLM
                     ├─ 3. auditor_response.j2              ← primary LLM (~800 tokens)
                     ├─ 4. Expanded retrieval if "Insufficient Info"  ← primary LLM
                     ├─ 5. _run_self_critique()             ← critique LLM (~600 tokens)
                     ├─ 6. auditor_revision.j2 (if score low) ← primary LLM (~800 tokens each)
                     ├─ 7. _verify_citations()             ← no LLM (regex)
                     ├─ 8. _cross_llm_critique()           ← judge LLM (~1000 tokens)
                     └─ 9. Confidence decomposition        ← no LLM (arithmetic)
                          │
                          ▼
                    audit_results.json (one entry per row)
                          │
                          ▼
                    validate_audit.py (optional)
                          ├─ CrossEncoder vs expert answer  ← no LLM
                          └─ LLM-as-judge vs expert answer  ← judge LLM
                               │
                               ▼
                         val_metrics_<ts>_<judge>.csv
```

**Token cost per control (approximate):**

| Mode | LLM calls | Tokens/row |
|---|---|---|
| LITE (answer only, no critique) | 2 (HyDE + answer) | ~800 |
| + self-critique (primary LLM) | 3 | ~1 400 |
| + critique-revise 1 revision | 5 | ~2 600 |
| + cross-LLM check | +1 | ~+1 000 |
| + query decomposition | +1 | ~+300 |
| Full pipeline (all enabled, 2 revisions) | up to 8 | ~4 000–5 000 |

---
