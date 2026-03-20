import os
import json
import time
import hashlib
from datetime import datetime, timezone
from config import CONFIG, PROJECT_ROOT
from rcm_engine import RcmAuditor
import pandas as pd


def _config_snapshot() -> dict:
    def _make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serialisable(i) for i in obj]
        if hasattr(obj, '__fspath__'):
            return str(obj)
        return obj
    return _make_serialisable(dict(CONFIG))


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_documents(folder: str) -> dict:
    hashes = {}
    if not os.path.exists(folder):
        return hashes
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            try:
                hashes[fname] = _hash_file(fpath)
            except Exception as e:
                hashes[fname] = f"error: {e}"
    return hashes


def main():
    print("Starting Audit Process...")

    run_start = datetime.now(timezone.utc)
    run_timestamp = run_start.strftime("%Y%m%dT%H%M%SZ")
    print(f"Run ID: {run_timestamp}")

    # Create per-run output folder
    run_dir = os.path.join(PROJECT_ROOT, "outputs", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output folder: {run_dir}")

    # Hash source documents for audit trail
    docs_folder = CONFIG['paths']['documents_folder']
    regs_folder = str(PROJECT_ROOT / "regulations")
    doc_hashes = _hash_documents(docs_folder)
    reg_hashes = _hash_documents(regs_folder)
    print(f"Document hashes computed: {len(doc_hashes)} client docs, {len(reg_hashes)} regulation docs.")

    auditor = RcmAuditor()
    print("Initializing RAG Engine (this may take a moment)...")
    auditor.initialize_rag()

    input_csv = CONFIG['paths']['input_csv']
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found.")
        return

    print(f"Reading input from {input_csv}...")
    df = None
    for encoding in ('utf-8', 'utf-8-sig', 'windows-1252', 'latin-1'):
        try:
            df = pd.read_csv(input_csv, sep=';', encoding=encoding)
            break
        except UnicodeDecodeError:
            print(f"Encoding {encoding} failed, trying next...")
        except Exception as e:
            print(f"Error reading CSV with {encoding}: {e}")
            return

    if df is None:
        print("Error: Could not read input CSV with any supported encoding.")
        return

    # Apply Tier filtering
    target_tier = str(CONFIG.get('filtering', {}).get('tier', 'all')).strip().lower()
    if target_tier != 'all':
        if 'Tier (1/2/3)' in df.columns:
            df['Tier (1/2/3)'] = df['Tier (1/2/3)'].astype(str).str.strip().str.lower()
            df = df[df['Tier (1/2/3)'] == target_tier]
            print(f"Filtered for Tier: {target_tier}. Remaining rows: {len(df)}")
        else:
            print("Warning: 'Tier (1/2/3)' column not found. Skipping tier filtering.")

    total_rows = len(df)
    print(f"Processing {total_rows} rows...")

    results = []
    for i, (idx, row) in enumerate(df.iterrows()):
        print(f"Processing row {i + 1}/{total_rows} (CSV row {idx + 2})...")
        try:
            res = auditor.process_row(row.to_dict())
            results.append(res)
        except Exception as e:
            print(f"Error on row {i + 1}: {e}")
            err_row = row.to_dict()
            err_row['AI_Answer'] = f"Error: {e}"
            err_row['Compliance_Verdict'] = "Insufficient Info"
            err_row['Validation_Score'] = 0
            err_row['Confidence_Score'] = 0.0
            results.append(err_row)
        time.sleep(1)

    # --- Save audit_results.json ---
    output_json = os.path.join(run_dir, "audit_results.json")
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Audit results saved to {output_json}")
    except Exception as e:
        print(f"Error saving results: {e}")
        return

    # --- Save run_manifest.json ---
    llm_settings = CONFIG.get('llm_settings', {})
    provider = llm_settings.get('provider', 'unknown')
    model = llm_settings.get(provider, {}).get('model', 'unknown')
    manifest = {
        "run_id":                    run_timestamp,
        "timestamp_utc":             run_start.isoformat(),
        "llm_provider":              provider,
        "llm_model":                 model,
        "tier_filter":               target_tier,
        "total_controls_processed":  len(results),
        "document_hashes":           doc_hashes,
        "regulation_hashes":         reg_hashes,
        "results_file":              output_json,
        "config_snapshot":           _config_snapshot(),
    }
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
        print(f"Run manifest saved to {manifest_path}")
    except Exception as e:
        print(f"Warning: Could not save run manifest: {e}")

    # --- Save flagged_for_review.json ---
    flag_score_threshold = CONFIG.get('audit_trail', {}).get('flag_score_threshold', 6)
    confidence_threshold = float(CONFIG.get('validation', {}).get('confidence_threshold', 60.0))
    flagged = []
    for r in results:
        reasons = []
        score = r.get('Validation_Score', 0)
        confidence = float(r.get('Confidence_Score', 100.0))
        verdict = r.get('Compliance_Verdict', '')
        cross_hallucinated = r.get('Cross_LLM_Hallucinated')

        if score < flag_score_threshold:
            reasons.append(f"Self-critique score {score} < threshold {flag_score_threshold}")
        if confidence < confidence_threshold:
            reasons.append(f"Confidence score {confidence:.1f} < threshold {confidence_threshold}")
        if verdict == 'Insufficient Info':
            reasons.append("Verdict is Insufficient Info")
        if cross_hallucinated is True:
            reasons.append("Cross-LLM critique detected potential hallucination")

        if reasons:
            flagged.append({
                "control_reference":    r.get('Control Reference', 'Unknown'),
                "compliance_verdict":   verdict,
                "validation_score":     score,
                "confidence_score":     confidence,
                "cross_llm_hallucinated": cross_hallucinated,
                "cross_llm_concerns":   r.get('Cross_LLM_Concerns', ''),
                "flag_reasons":         reasons,
            })

    flagged_path = os.path.join(run_dir, "flagged_for_review.json")
    try:
        with open(flagged_path, 'w', encoding='utf-8') as f:
            json.dump(flagged, f, indent=4, ensure_ascii=False)
        print(f"Flagged {len(flagged)}/{len(results)} controls for human review → {flagged_path}")
    except Exception as e:
        print(f"Warning: Could not save flagged review file: {e}")

    # --- Run validation and save val_metrics CSV into the same run folder ---
    print("\nStarting validation step...")
    try:
        from validate_audit import validate_audit
        validate_audit(run_folder=run_dir)
    except Exception as e:
        print(f"Warning: Validation step failed: {e}")

    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
