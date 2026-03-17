import pandas as pd
import json
import os
import time
from config import CONFIG
from sentence_transformers import CrossEncoder
import math
from deep_translator import GoogleTranslator
import re
from llm_factory import get_llm

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
    
    # Ensure Control Reference is string and clean
    if 'Control Reference' in df_ai.columns:
        df_ai['Control Reference'] = df_ai['Control Reference'].astype(str).str.strip()
    else:
        print("Error: 'Control Reference' column missing in AI results.", flush=True)
        return

    # 2. Load Expert Answers
    expert_csv = CONFIG['paths']['expert_answers_csv']
    if not os.path.exists(expert_csv):
        print(f"Error: {expert_csv} not found.", flush=True)
        return

    print(f"Loading Expert answers from {expert_csv}...", flush=True)
    try:
        # Based on inspection, separator is ';'
        df_expert = pd.read_csv(expert_csv, sep=';', encoding='latin-1') 
    except Exception as e:
        print(f"Error reading expert CSV: {e}", flush=True)
        return

    # Ensure Control Reference is string and clean
    if 'Control Reference' in df_expert.columns:
        df_expert['Control Reference'] = df_expert['Control Reference'].astype(str).str.strip()
    else:
        print("Error: 'Control Reference' column missing in expert CSV.", flush=True)
        return

    # 3. Merge DataFrames
    print("Merging data...", flush=True)
    # Select relevant columns from expert to avoid clutter
    cols_to_keep = ['Control Reference', 'Answers based on Clients data', 'Tier (1/2/3)']
    
    # Try to keep Tier column if it exists for filtering
    existing_cols = [c for c in cols_to_keep if c in df_expert.columns]
    df_expert_clean = df_expert[existing_cols].dropna(subset=['Answers based on Clients data'])
    
    # Apply Tier filtering if Tier is present
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
            print("Warning: 'Tier (1/2/3)' column not found in either dataset. Skipping filtering.", flush=True)


    # Avoid duplicated Tier column after merge
    if 'Tier (1/2/3)' in df_ai.columns and 'Tier (1/2/3)' in df_expert_clean.columns:
        df_expert_clean = df_expert_clean.drop(columns=['Tier (1/2/3)'])

    # Drop Answers from df_ai so that we only have the pristine expert version
    if 'Answers based on Clients data' in df_ai.columns:
        df_ai = df_ai.drop(columns=['Answers based on Clients data'])

    merged_df = pd.merge(df_ai, df_expert_clean, on='Control Reference', how='inner')
    
    print(f"Merged {len(merged_df)} rows (Intersection of AI and Expert data).", flush=True)

    print("Loading CrossEncoder model (this may take a moment)...", flush=True)
    try:
        model = CrossEncoder('cross-encoder/stsb-roberta-base')
    except Exception as e:
        print(f"Error initializing CrossEncoder: {e}", flush=True)
        return

    print("Initializing Google Translator...", flush=True)
    translator = GoogleTranslator(source='auto', target='en')

    print("Calculating similarity metrics and LLM judgments...", flush=True)
    
    # Initialize metrics columns
    merged_df['Cross_Encoder_Score'] = 0.0
    # merged_df['Comparison_Score'] = 0.0
    merged_df['Validation_Score_LLM_as_judge'] = 0.0
    merged_df['Stability_Score'] = 0.0
    merged_df['Drift_Resistance_Score'] = 0.0
    merged_df['Guardrail_Effectiveness_Score'] = 0.0
    merged_df['Reasoning_LLM_as_judge'] = ""
    
    llm = get_llm()

    def get_llm_judgment(expert_ans, ai_ans, retries=3):
        prompt = f"""You are an expert auditor evaluating an AI's answer against a Ground Truth expert answer.
Evaluate the AI answer across four dimensions and give a score from 0 to 100 for each (100 is best):
1. Accuracy: How accurately the AI answer captures the factual essence of the Ground Truth.
2. Stability: Assess if the AI reasoning seems robust, consistent and logically stable.
3. Drift Resistance: How well the AI answer stays grounded in the context without hallucinating external information or drifting.
4. Guardrail Effectiveness: How well the AI maintains a professional, objective auditing tone and avoids inappropriate/prohibited content.

Ground Truth Expert Answer:
{expert_ans}

AI Generated Answer:
{ai_ans}

Provide your response in strictly valid JSON format with exactly these keys: "accuracy_score", "stability_score", "drift_resistance_score", "guardrail_score" (all integers 0-100), and "reasoning" (a brief string explaining the scores)."""
        for attempt in range(retries):
            try:
                response = llm.invoke(prompt)
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                result = json.loads(content)
                return {
                    "accuracy": float(result.get("accuracy_score", 0)),
                    "stability": float(result.get("stability_score", 0)),
                    "drift_resistance": float(result.get("drift_resistance_score", 0)),
                    "guardrail": float(result.get("guardrail_score", 0)),
                    "reasoning": str(result.get("reasoning", "No reasoning provided."))
                }
            except Exception as e:
                err_str = str(e)
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"Error during LLM judgment after {retries} attempts: {e}", flush=True)
                    return {"accuracy": 0.0, "stability": 0.0, "drift_resistance": 0.0, "guardrail": 0.0, "reasoning": f"Error: {e}"}
    
    output_path = CONFIG['paths']['validation_report_csv']

    
    for idx, row in merged_df.iterrows():
        question = row.get('Design Effectiveness Assessment', 'N/A')
        ai_ans = row.get('AI_Answer', 'N/A')
        expert_ans = row.get('Answers based on Clients data', 'N/A')
        
        # Translate to unified English
        try:
            ai_ans_en = translator.translate(ai_ans)
            expert_ans_en = translator.translate(expert_ans)
        except Exception as e:
            print(f"Translation Error row {idx}: {e}", flush=True)
            ai_ans_en, expert_ans_en = ai_ans, expert_ans # Fallback to original
        
        # 1. Semantic Similarity (CrossEncoder)
        try:
            # CrossEncoder expects a list of pairs for inference
            # stsb-roberta-base outputs a score from 0.0 to 1.0 representing semantic similarity
            raw_score = float(model.predict([ai_ans_en, expert_ans_en]))
            # Convert to [0, 100] percentage and cap just in case
            cross_encoder_score = max(0.0, min(100.0, raw_score * 100.0))
        except Exception as e:
            print(f"Error calculating cross-encoder similarity for row {idx}: {e}")
            cross_encoder_score = 0.0
        
        # 2. LLM AS A JUDGE (Use original untranslated text)
        llm_results = get_llm_judgment(expert_ans, ai_ans)
        
        merged_df.loc[idx, 'Cross_Encoder_Score'] = cross_encoder_score
        merged_df.loc[idx, 'Validation_Score_LLM_as_judge'] = llm_results['accuracy']
        merged_df.loc[idx, 'Stability_Score'] = llm_results['stability']
        merged_df.loc[idx, 'Drift_Resistance_Score'] = llm_results['drift_resistance']
        merged_df.loc[idx, 'Guardrail_Effectiveness_Score'] = llm_results['guardrail']
        merged_df.loc[idx, 'Reasoning_LLM_as_judge'] = llm_results['reasoning']
        merged_df['Expert_Answer_EN'] = expert_ans_en

        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(merged_df)}...", flush=True)
            try:
                merged_df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
            except PermissionError:
                alt_path = output_path.replace('.csv', '_new.csv')
                print(f"Warning: {output_path} is locked. Saving intermediate to {alt_path}", flush=True)
                merged_df.to_csv(alt_path, index=False, encoding='utf-8-sig', sep=';')
            time.sleep(1) # Polite delay
            
    try:
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
        print(f"Validation complete. Report saved to {output_path}", flush=True)
    except PermissionError:
        alt_path = output_path.replace('.csv', '_new.csv')
        merged_df.to_csv(alt_path, index=False, encoding='utf-8-sig', sep=';')
        print(f"Validation complete. Original file was locked, report saved to {alt_path}", flush=True)

if __name__ == "__main__":
    validate_audit()
