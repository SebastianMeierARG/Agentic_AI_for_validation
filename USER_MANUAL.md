# User Manual — IFRS 9 Agentic Audit Validation Tool

**Version:** Current (March 2026)
**Audience:** Audit professionals, credit risk analysts, model validators

---

## Table of Contents

1. [What This Tool Does](#1-what-this-tool-does)
2. [Before You Start — Know Your Resources](#2-before-you-start--know-your-resources)
3. [Validation Strategy A — Cross-LLM as Judge (Recommended for Regulatory Use)](#3-validation-strategy-a--cross-llm-as-judge)
4. [Validation Strategy B — Self-Critique Only (Faster, Less Compliant)](#4-validation-strategy-b--self-critique-only)
5. [Step-by-Step: Running a Full Audit](#5-step-by-step-running-a-full-audit)
6. [Understanding Your Results](#6-understanding-your-results)
7. [The Output Folder — What Each File Means](#7-the-output-folder--what-each-file-means)
8. [Configuring the Tool](#8-configuring-the-tool)
9. [Using the Interactive Notebook for Spot Checks](#9-using-the-interactive-notebook-for-spot-checks)
10. [Troubleshooting](#10-troubleshooting)
11. [Quick Reference Card](#11-quick-reference-card)

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
5. Has a second, independent AI model check every claim for accuracy
6. Flags any control where confidence is low, for a human auditor to review

### What IFRS 9 / ECL means in plain terms

The ECL can be interpret as the bank's best estimate of "how much of what we lent out we probably won't get back." IFRS 9 requires this estimate to be calculated using a specific methodology, approved by senior governance, documented in formal policies, and regularly validated. The audit checks whether all of that is actually happening and properly evidenced.

### What the output looks like

For each control on your checklist, the tool produces:

- **AI Answer** — A structured written response citing specific pages from your documents
- **Compliance Verdict** — Compliant / Non-Compliant / Partial / Insufficient Info
- **Confidence Score** — 0 to 100; below 60 means the tool is uncertain and flags the control for human review
- **Validation Score** — 0 to 10; how well the answer is supported by the documents
- **Hallucination Flag** — whether the independent AI checker found any claims not backed by documents

At the end you also receive a validation report comparing the AI's answers to expert reference answers, scored on accuracy, stability, and consistency.

---

## 2. Before You Start — Know Your Resources

Answer these three questions before you choose how to run the tool. Your answers will determine which approach gives you the best result for your situation.

---

**Question 1: How much time do I have?**

- **A few hours for a quick draft** → Use LITE mode in the notebook (Section 9). Skips independent checking, faster output.
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

## 3. Validation Strategy A — Cross-LLM as Judge

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

## 4. Validation Strategy B — Self-Critique Only

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

In the Jupyter notebook (`notebooks/interactive_audit.ipynb`), set the following at the top of Section 0:

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

## 5. Step-by-Step: Running a Full Audit

### Prerequisites

Before running the tool, confirm:

- [ ] Python 3.10 or later is installed
- [ ] You have run `pip install -r requirements.txt` at least once
- [ ] Your `.env` file exists in the project root and contains at least `OPENAI_API_KEY`
- [ ] Your client PDF documents are placed in the `documents/` folder
- [ ] Your audit control checklist is saved as `inputs/rcm_input.csv`
- [ ] (Optional) Expert reference answers are in `inputs/rcm_expert_answer.csv`

---

### Path 1: Full pipeline — audit and validate in one command

This is the recommended approach. It runs the audit, then immediately validates the results.

```bash
python src/run_audit.py
```

This single command:
1. Reads all documents in `documents/` and builds the search index
2. Processes every control in `rcm_input.csv`
3. Generates audit answers with evidence citations
4. Runs the independent judge model for hallucination checks
5. Saves all results into a timestamped folder under `outputs/`
6. Automatically runs the validation step against expert answers
7. Saves the final `val_metrics_*.csv` in the same folder

When it finishes, you will see a message like:
```
All outputs saved to: outputs/20260320T143022Z/
```

---

### Path 2: Audit only, then validate separately

If you want to inspect the raw audit results before running validation:

```bash
# Step 1: Run the audit
python src/run_audit.py

# Step 2: Run validation separately (auto-detects the latest run folder)
python src/validate_audit.py
```

---

### Path 3: Run the interactive notebook for spot checks

For checking individual controls before committing to a full run:

```bash
jupyter notebook notebooks/interactive_audit.ipynb
```

See Section 9 of this manual for how to use the notebook.

---

### Filtering by tier

If you only want to audit Tier 1 controls (the highest-risk controls), edit `config.yaml`:

```yaml
filtering:
  tier: '1'    # options: 'all', '1', '2', '3'
```

Change this before running. Tier 1 is the default.

---

### What to expect during a run

The tool prints progress for each control as it processes:

```
Run ID: 20260320T143022Z
Output folder: outputs/20260320T143022Z/
Initializing RAG Engine...
Processing 32 rows...
Processing row 1/32 (CSV row 2)...
Processing row 2/32 (CSV row 3)...
...
Audit results saved to outputs/20260320T143022Z/audit_results.json
Flagged 5/32 controls for human review
Starting validation step...
Judge LLM: Ollama (local) / llama3.1:8b
...
Validation complete. Report saved to outputs/20260320T143022Z/val_metrics_20260320T143022Z_ollama_llama3.1-8b.csv
```

A full run of 32 Tier 1 controls typically takes:
- **With Groq as judge:** 15–25 minutes
- **With Ollama on CPU:** 60–120 minutes
- **With Ollama on GPU:** 20–40 minutes

---

## 6. Understanding Your Results

### Compliance Verdict

Each control receives one of four verdicts:

| Verdict | Meaning |
|---|---|
| **Compliant** | The documents clearly evidence that the requirement is met |
| **Non-Compliant** | The documents show the requirement is not met, or evidence contradicts it |
| **Partial** | Some aspects are evidenced but others are missing or unclear |
| **Insufficient Info** | The documents do not contain enough information to make a determination |

An "Insufficient Info" verdict does not mean the bank is non-compliant — it means the documents provided to the tool do not allow a conclusion to be drawn. Additional documents may be needed.

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

### Confidence Score (0–100)

This score reflects how certain the tool is about its own answer. It is a blend of the self-critique score and the cross-LLM hallucination check.

- **80–100:** High confidence. The answer is well-supported and the independent checker found no issues.
- **60–79:** Moderate confidence. Review the answer but it is likely reliable.
- **Below 60:** Low confidence. The control is automatically flagged for human review. This does not mean the answer is wrong — it means a human auditor should verify it.

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

The overall **Weighted Accuracy Score** multiplies each control's accuracy score by its tier weight (Tier 1 = 3×, Tier 2 = 2×, Tier 3 = 1×) and averages the result. This gives higher-risk controls more influence on the summary metric.

---

### Flagged Controls

Any control that meets one or more of these conditions is written to `flagged_for_review.json` and marked for human attention:

- Self-critique score below 6 out of 10
- Confidence score below 60 out of 100
- Verdict of "Insufficient Info"
- Independent checker detected a claim not supported by the documents

Flagged controls are not failures — they are the tool's way of saying "a human auditor should look at this one." In a well-functioning audit, you would expect 10–30% of controls to be flagged for review.

---

## 7. The Output Folder — What Each File Means

Each run creates its own folder inside `outputs/`, named by the timestamp of when the run started:

```
outputs/
  20260320T143022Z/
    audit_results.json
    run_manifest.json
    flagged_for_review.json
    val_metrics_20260320T143022Z_ollama_llama3.1-8b.csv
```

---

### audit_results.json

The main output. Contains one entry per control with:

- The full AI-written answer
- Page and document citations for each claim
- Compliance verdict
- Self-critique score and reasoning
- Hallucination flag from the independent checker
- Confidence score

This is the file you would use to review individual control answers.

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

A filtered list of only the controls that require human attention, with the specific reason each was flagged. Use this as your starting point for manual review — you do not need to read all 32 or 68 controls; focus on the flagged ones first.

---

### val_metrics_*.csv

The validation report. The filename includes the UTC timestamp and the name of the judge model used (e.g. `ollama_llama3.1-8b`), so you can always trace a report back to the exact model that produced it.

Open this file in Excel or any spreadsheet tool. Key columns to review:

- `Control Reference` — the control ID
- `Compliance_Verdict` — Compliant / Non-Compliant / Partial / Insufficient Info
- `Validation_Score_LLM_as_judge` — overall accuracy score vs expert (0–100)
- `Cross_Encoder_Score` — semantic similarity vs expert (0–100)
- `Reasoning_LLM_as_judge` — the judge model's written explanation of the score
- `Answers based on Clients data` — the expert reference answer (for comparison)

---

## 8. Configuring the Tool

All settings are in `config.yaml` in the project root. Open it with any text editor. The most commonly changed settings are:

### Which controls to run

```yaml
filtering:
  tier: '1'    # change to '2', '3', or 'all'
```

### Using your own language documents

If your client documents are in a language other than Spanish:

```yaml
rag_settings:
  document_language: "Portuguese"   # or "English", "French", "German", etc.
```

The retrieval and reranking systems adapt automatically.

### Turning off regulation documents

If you only want the tool to search client documents (not the EBA/IFRS 9 regulation PDFs):

```yaml
rag_settings:
  use_regulations: false
```

This saves approximately 1,500 tokens per control and speeds up the run.

### Changing which Ollama model to use

```yaml
judge_llm:
  ollama_model: "llama3.1:8b"   # must match a model you have pulled with 'ollama pull'
```

### Enabling or disabling validation layers

```yaml
validation:
  enable_self_critique: true          # judge LLM scores each answer 0-10
  enable_cross_llm_critique: true     # independent hallucination check
  confidence_threshold: 60.0          # controls below this are flagged
```

### Changing the confidence threshold for flagging

If you want to flag more controls for review (be more cautious):

```yaml
validation:
  confidence_threshold: 70.0    # was 60.0 — now flags more controls
```

---

## 9. Using the Interactive Notebook for Spot Checks

The notebook (`notebooks/interactive_audit.ipynb`) lets you test individual controls without running the full pipeline. This is useful when:

- You want to verify the tool is retrieving the right document sections
- You want to check one specific control before committing to a full run
- You are adjusting settings and want to see the effect quickly

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

At the top of Section 0, you control token usage:

```python
LITE_MODE = True    # True = answer only (~800 tokens/row)
                    # False = answer + self-critique + cross-LLM (~2,500 tokens/row)
```

Use LITE mode when exploring. Switch to FULL mode when you need the complete validation.

---

## 10. Troubleshooting

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

### "Error: Extra data" in val_metrics reasoning column

The judge model returned text after its JSON response. This is handled automatically since version March 2026. If you see it in older runs, re-run `validate_audit.py`.

### Run is very slow

If Ollama is being used as judge on a CPU-only machine, each control check takes 15–30 seconds. For 32 controls with self-critique + cross-LLM, expect 30–60 minutes. Options:
- Use Groq instead (add `GROQ_API_KEY` to `.env`)
- Disable self-critique: `enable_self_critique: false` in `config.yaml`
- Use a smaller Ollama model: `ollama pull llama3.2` (2 GB, faster)

### Groq quota exhausted mid-run

You will see a message like `Provider exhausted. Trying next...` — the tool automatically switches to Together AI and continues. No action needed. If you also see Together AI failing, add an Ollama fallback (see Option A3).

### Results look irrelevant — wrong document sections retrieved

Open the notebook and run Section 1 (RAG Debug) with your control's question. If the retrieved chunks are not from the right documents, try:
- Lowering `retrieval_score_threshold` in `config.yaml` (e.g. from 1.8 to 1.5) for stricter filtering
- Increasing `client_top_k` to retrieve more chunks
- Checking that your PDFs are properly placed in `documents/` (not in subfolders)

### FAISS index seems stale after adding new documents

Delete the index folders and let the tool rebuild them:
```bash
# Windows
rmdir /s /q faiss_index_client
rmdir /s /q faiss_index_regs

# Mac/Linux
rm -rf faiss_index_client faiss_index_regs
```
The next run will rebuild the indices from scratch.

---

## 11. Quick Reference Card

### Commands

| Task | Command |
|---|---|
| Full audit + validation | `python src/run_audit.py` |
| Validation only (latest run) | `python src/validate_audit.py` |
| Client policy summary | `python src/run_summary.py` |
| Interactive notebook | `jupyter notebook notebooks/interactive_audit.ipynb` |

### Key files to know

| File | What you edit / read |
|---|---|
| `config.yaml` | All settings: tier, language, models, thresholds |
| `.env` | API keys |
| `inputs/rcm_input.csv` | Your audit control checklist |
| `inputs/rcm_expert_answer.csv` | Expert reference answers (for validation) |
| `documents/` | Put your client PDFs here |
| `outputs/<timestamp>/` | All results from a run |

### config.yaml most-used settings

| Setting | Default | What it does |
|---|---|---|
| `filtering.tier` | `'1'` | Which tier to audit (`'all'`, `'1'`, `'2'`, `'3'`) |
| `rag_settings.document_language` | `Spanish` | Language of your client documents |
| `rag_settings.use_regulations` | `true` | Include EBA regulation docs in search |
| `validation.enable_self_critique` | `true` | Score each answer 0–10 (uses judge LLM) |
| `validation.enable_cross_llm_critique` | `true` | Independent hallucination check |
| `validation.confidence_threshold` | `60.0` | Below this → flagged for human review |
| `judge_llm.ollama_model` | `llama3.1:8b` | Which local Ollama model to use |

### Judge LLM fallback order

The tool tries these providers in order, using the first one available:

1. **Ollama** (local) — if installed and running
2. **Groq** — if `GROQ_API_KEY` is in `.env`
3. **Together AI** — if `TOGETHER_API_KEY` is in `.env`
4. **Secondary provider** (Google or OpenAI, whichever is not primary)
5. **Primary LLM** — last resort

---

