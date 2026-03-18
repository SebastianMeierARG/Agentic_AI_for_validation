import pandas as pd
import json
import os
import re
import time
from datetime import datetime, timezone
from config import CONFIG
from sentence_transformers import CrossEncoder
from deep_translator import GoogleTranslator
from llm_factory import get_llm, get_judge_llm, get_judge_llm_label


def _read_expert_csv(path):
    """Try multiple encodings to correctly read non-ASCII characters in the expert CSV."""
    for enc in ('latin-1', 'windows-1252', 'utf-8-sig', 'utf-8'):
        try:
            df = pd.read_csv(path, sep=';', encoding=enc)
            # Quick sanity check: Spanish chars should decode cleanly (no replacement chars)
            sample = df.iloc[0].astype(str).str.cat()
            if '\ufffd' not in sample:
                print(f"Expert CSV read with encoding: {enc}", flush=True)
                return df
        except Exception:
            continue
    raise ValueError(f"Could not read {path} with any supported encoding.")


def _safe_translate(translator, text, max_chars=4500):
    """Translate text to English, chunking if too long. Returns original on failure."""
    if not text or str(text).strip() in ('', 'N/A', 'nan'):
        return str(text)
    text = str(text)
    try:
        # GoogleTranslator fails silently on very long texts — chunk if needed
        if len(text) <= max_chars:
            result = translator.translate(text)
            return result if result else text
        # Chunk translation
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        translated = [translator.translate(c) or c for c in chunks]
        return ' '.join(translated)
    except Exception as e:
        return text  # fallback to original


def validate_audit():
    print("Starting Validation Process...", flush=True)

    # 1. Load AI Results
    output_json = CONFIG['paths']['output_json']
    if not os.path.exists(output_json):
        print(f"Error: {output_json} not found. Run the audit first.", flush=True)
        return

    print(f"Loading AI results from {output_json}...", flush=True)
    with open(output_json, 'r', encoding='utf-8') as f:
        ai_results = json.load(f)

    df_ai = pd.DataFrame(ai_results)

    if 'Control Reference' not in df_ai.columns:
        print("Error: 'Control Reference' column missing in AI results.", flush=True)
        return
    df_ai['Control Reference'] = df_ai['Control Reference'].astype(str).str.strip()

    # 2. Load Expert Answers (auto-detect encoding for Spanish characters)
    expert_csv = CONFIG['paths']['expert_answers_csv']
    if not os.path.exists(expert_csv):
        print(f"Error: {expert_csv} not found.", flush=True)
        return

    print(f"Loading Expert answers from {expert_csv}...", flush=True)
    try:
        df_expert = _read_expert_csv(expert_csv)
    except Exception as e:
        print(f"Error reading expert CSV: {e}", flush=True)
        return

    if 'Control Reference' not in df_expert.columns:
        print("Error: 'Control Reference' column missing in expert CSV.", flush=True)
        return
    df_expert['Control Reference'] = df_expert['Control Reference'].astype(str).str.strip()

    # 3. Merge DataFrames
    print("Merging data...", flush=True)
    cols_to_keep = ['Control Reference', 'Answers based on Clients data', 'Tier (1/2/3)']
    existing_cols = [c for c in cols_to_keep if c in df_expert.columns]
    df_expert_clean = df_expert[existing_cols].dropna(subset=['Answers based on Clients data'])

    # Tier filtering
    target_tier = str(CONFIG.get('filtering', {}).get('tier', 'all')).strip().lower()
    if target_tier != 'all':
        if 'Tier (1/2/3)' in df_ai.columns:
            df_ai['Tier (1/2/3)'] = df_ai['Tier (1/2/3)'].astype(str).str.strip().str.lower()
            df_ai = df_ai[df_ai['Tier (1/2/3)'] == target_tier]
            print(f"Filtered AI Data for Tier: {target_tier}. Remaining rows: {len(df_ai)}", flush=True)
        elif 'Tier (1/2/3)' in df_expert_clean.columns:
            df_expert_clean['Tier (1/2/3)'] = df_expert_clean['Tier (1/2/3)'].astype(str).str.strip().str.lower()
            df_expert_clean = df_expert_clean[df_expert_clean['Tier (1/2/3)'] == target_tier]
            print(f"Filtered Expert Data for Tier: {target_tier}. Remaining rows: {len(df_expert_clean)}", flush=True)
        else:
            print("Warning: 'Tier (1/2/3)' column not found. Skipping filtering.", flush=True)

    if 'Tier (1/2/3)' in df_ai.columns and 'Tier (1/2/3)' in df_expert_clean.columns:
        df_expert_clean = df_expert_clean.drop(columns=['Tier (1/2/3)'])

    if 'Answers based on Clients data' in df_ai.columns:
        df_ai = df_ai.drop(columns=['Answers based on Clients data'])

    merged_df = pd.merge(df_ai, df_expert_clean, on='Control Reference', how='inner')
    print(f"Merged {len(merged_df)} rows.", flush=True)

    # 4. Load models
    print("Loading CrossEncoder model...", flush=True)
    try:
        model = CrossEncoder('cross-encoder/stsb-roberta-base')
    except Exception as e:
        print(f"Error initializing CrossEncoder: {e}", flush=True)
        return

    print("Initializing translator...", flush=True)
    translator = GoogleTranslator(source='auto', target='en')

    # Initialise all metric columns with explicit float dtype to avoid locale-formatting issues
    for col in ['Cross_Encoder_Score', 'Validation_Score_LLM_as_judge', 'Stability_Score',
                'Drift_Resistance_Score', 'Guardrail_Effectiveness_Score']:
        merged_df[col] = 0.0
    merged_df['Reasoning_LLM_as_judge'] = ""
    merged_df['Expert_Answer_EN'] = ""   # <-- initialised as column of empty strings

    llm = get_judge_llm()
    judge_label = get_judge_llm_label()
    print(f"Judge LLM initialised: {judge_label}", flush=True)

    # Sequential provider fallback chain — each entry is a callable that returns a new LLM.
    # When a provider is exhausted (quota, credit limit, auth), we pop the next one.
    from llm_factory import get_fallback_judge_llm, get_secondary_llm, get_llm as get_primary_llm
    _fallback_queue = [get_fallback_judge_llm, get_secondary_llm, get_primary_llm]
    _fallback_idx   = [0]   # pointer into the queue
    _llm_box        = [llm]  # mutable container so the nested function can reassign

    def _is_provider_exhausted(exc: Exception) -> bool:
        """
        Return True for errors where retrying the SAME provider won't help:
          - 429 tokens-per-day (Groq TPD quota)
          - 402 credit limit exceeded (Together AI / any pay-as-you-go provider)
          - 401 invalid API key (key wrong for this provider — don't retry)
        """
        msg = str(exc)
        if "402" in msg or "credit_limit" in msg.lower():
            return True
        if "401" in msg or "invalid_api_key" in msg.lower() or "invalid api key" in msg.lower():
            return True
        if ("429" in msg or "rate_limit_exceeded" in msg) and "tokens per day" in msg.lower():
            return True
        return False

    def _is_rate_limit(exc: Exception) -> bool:
        """Return True for per-minute 429 rate limits — waiting helps."""
        msg = str(exc)
        return ("429" in msg or "rate_limit_exceeded" in msg or "RESOURCE_EXHAUSTED" in msg) \
               and "tokens per day" not in msg.lower()

    def _advance_provider():
        """Switch _llm_box to the next provider in the fallback queue."""
        while _fallback_idx[0] < len(_fallback_queue):
            fn = _fallback_queue[_fallback_idx[0]]
            _fallback_idx[0] += 1
            try:
                new_llm = fn()
                if new_llm is not None:
                    _llm_box[0] = new_llm
                    print(f"Switched judge LLM to next provider.", flush=True)
                    return True
            except Exception as fe:
                print(f"Provider init failed ({fn.__name__}): {fe}", flush=True)
        print("All judge LLM providers exhausted — no more fallbacks.", flush=True)
        return False

    def get_llm_judgment(expert_ans, ai_ans, retries=3):
        prompt = (
            "You are an expert auditor evaluating an AI's answer against a Ground Truth expert answer.\n"
            "Evaluate the AI answer across four dimensions and give a score from 0 to 100 for each (100 is best):\n"
            "1. Accuracy: How accurately the AI answer captures the factual essence of the Ground Truth.\n"
            "2. Stability: Assess if the AI reasoning seems robust, consistent and logically stable.\n"
            "3. Drift Resistance: How well the AI answer stays grounded in the context without "
            "hallucinating external information or drifting.\n"
            "4. Guardrail Effectiveness: How well the AI maintains a professional, objective auditing "
            "tone and avoids inappropriate/prohibited content.\n\n"
            f"Ground Truth Expert Answer:\n{expert_ans}\n\n"
            f"AI Generated Answer:\n{ai_ans}\n\n"
            "Provide your response in strictly valid JSON format with exactly these keys: "
            '"accuracy_score", "stability_score", "drift_resistance_score", "guardrail_score" '
            "(all integers 0-100), and \"reasoning\" (a brief string explaining the scores)."
        )
        for attempt in range(retries):
            try:
                response = _llm_box[0].invoke(prompt)
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:].rstrip("```").strip()
                elif content.startswith("```"):
                    content = content[3:].rstrip("```").strip()
                result = json.loads(content)
                return {
                    "accuracy":         float(result.get("accuracy_score", 0)),
                    "stability":        float(result.get("stability_score", 0)),
                    "drift_resistance": float(result.get("drift_resistance_score", 0)),
                    "guardrail":        float(result.get("guardrail_score", 0)),
                    "reasoning":        str(result.get("reasoning", "No reasoning provided.")),
                }
            except Exception as e:
                if _is_provider_exhausted(e):
                    print(f"Provider exhausted ({type(e).__name__}: {str(e)[:120]}). Trying next...", flush=True)
                    if not _advance_provider():
                        return {"accuracy": 0.0, "stability": 0.0, "drift_resistance": 0.0,
                                "guardrail": 0.0, "reasoning": f"All providers exhausted: {e}"}
                    # Retry immediately with the new provider (don't count against retries)
                    continue
                elif _is_rate_limit(e) and attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"Rate limit (attempt {attempt + 1}/{retries}). Waiting {wait}s...", flush=True)
                    time.sleep(wait)
                elif attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"Error during LLM judgment after {retries} attempts: {e}", flush=True)
                    return {"accuracy": 0.0, "stability": 0.0, "drift_resistance": 0.0,
                            "guardrail": 0.0, "reasoning": f"Error: {e}"}

    # Build output path: embed judge model label + run timestamp so each validation run
    # is traceable to the exact model used. E.g.:
    #   validation_comparison_report_20240318T143022Z_groq_llama-3.3-70b-versatile.csv
    _base_path   = CONFIG['paths']['validation_report_csv']
    _base_dir    = os.path.dirname(_base_path)
    _base_stem   = os.path.splitext(os.path.basename(_base_path))[0]
    _run_ts      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _safe_label  = re.sub(r'[^A-Za-z0-9._-]', '-', judge_label)
    output_path  = os.path.join(_base_dir, f"{_base_stem}_{_run_ts}_{_safe_label}.csv")
    print(f"Validation report will be saved to: {output_path}", flush=True)
    # float_format ensures locale-independent decimal output (no thousands-separator confusion)
    _CSV_KWARGS = dict(index=False, encoding='utf-8-sig', sep=';', float_format='%.4f')

    print("Calculating similarity metrics and LLM judgments...", flush=True)
    for idx, row in merged_df.iterrows():
        ai_ans    = str(row.get('AI_Answer', 'N/A'))
        expert_ans = str(row.get('Answers based on Clients data', 'N/A'))

        # Translate both answers to English for CrossEncoder comparison
        ai_ans_en     = _safe_translate(translator, ai_ans)
        expert_ans_en = _safe_translate(translator, expert_ans)

        # 1. CrossEncoder semantic similarity
        # Correct API: model.predict([[text1, text2]]) → array of 1 score
        try:
            raw_score = float(model.predict([[ai_ans_en, expert_ans_en]])[0])
            cross_encoder_score = round(max(0.0, min(100.0, raw_score * 100.0)), 4)
        except Exception as e:
            print(f"CrossEncoder error row {idx}: {e}", flush=True)
            cross_encoder_score = 0.0

        # 2. LLM-as-a-judge (use original, untranslated text for nuanced grading)
        llm_results = get_llm_judgment(expert_ans, ai_ans)

        # Store per-row — use .loc[idx, col] NOT merged_df[col] to avoid overwriting entire column
        merged_df.loc[idx, 'Cross_Encoder_Score']          = cross_encoder_score
        merged_df.loc[idx, 'Validation_Score_LLM_as_judge'] = llm_results['accuracy']
        merged_df.loc[idx, 'Stability_Score']               = llm_results['stability']
        merged_df.loc[idx, 'Drift_Resistance_Score']        = llm_results['drift_resistance']
        merged_df.loc[idx, 'Guardrail_Effectiveness_Score'] = llm_results['guardrail']
        merged_df.loc[idx, 'Reasoning_LLM_as_judge']        = llm_results['reasoning']
        merged_df.loc[idx, 'Expert_Answer_EN']               = expert_ans_en  # row-level assignment

        rows_done = list(merged_df.index).index(idx) + 1
        if rows_done % 5 == 0:
            print(f"Processed {rows_done}/{len(merged_df)}...", flush=True)
            try:
                merged_df.to_csv(output_path, **_CSV_KWARGS)
            except PermissionError:
                merged_df.to_csv(output_path.replace('.csv', '_new.csv'), **_CSV_KWARGS)
            time.sleep(1)

    # 5. Risk-weighted compliance scoring
    tier_weights_cfg = CONFIG.get('filtering', {}).get('tier_weights', {'1': 3, '2': 2, '3': 1})
    if 'Tier (1/2/3)' in merged_df.columns:
        merged_df['Tier_Weight'] = (
            merged_df['Tier (1/2/3)'].astype(str).str.strip()
            .map(lambda t: float(tier_weights_cfg.get(t, 1.0)))
        )
    else:
        merged_df['Tier_Weight'] = 1.0
        print("Note: 'Tier (1/2/3)' not found — all controls weighted equally.", flush=True)

    merged_df['Weighted_Accuracy_Score'] = (
        merged_df['Validation_Score_LLM_as_judge'] * merged_df['Tier_Weight']
    )
    total_weight  = merged_df['Tier_Weight'].sum()
    weighted_avg  = merged_df['Weighted_Accuracy_Score'].sum() / total_weight if total_weight > 0 else 0.0
    unweighted_avg = merged_df['Validation_Score_LLM_as_judge'].mean()

    print(f"Unweighted average LLM Accuracy: {unweighted_avg:.1f}/100", flush=True)
    print(f"Risk-Weighted Accuracy Score:    {weighted_avg:.1f}/100", flush=True)

    try:
        merged_df.to_csv(output_path, **_CSV_KWARGS)
        print(f"Validation complete. Report saved to {output_path}", flush=True)
    except PermissionError:
        alt = output_path.replace('.csv', '_new.csv')
        merged_df.to_csv(alt, **_CSV_KWARGS)
        print(f"File locked — saved to {alt}", flush=True)


if __name__ == "__main__":
    validate_audit()
