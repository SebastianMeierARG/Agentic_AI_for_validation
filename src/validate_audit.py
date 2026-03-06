import pandas as pd
import json
import os
import time
from config import CONFIG
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
        if 'Tier (1/2/3)' in df_expert_clean.columns:
            df_expert_clean['Tier (1/2/3)'] = df_expert_clean['Tier (1/2/3)'].astype(str).str.strip().str.lower()
            df_expert_clean = df_expert_clean[df_expert_clean['Tier (1/2/3)'] == target_tier]
            print(f"Filtered Expert Data for Tier: {target_tier}. Remaining rows: {len(df_expert_clean)}", flush=True)
        else:
            print("Warning: 'Tier (1/2/3)' column not found in expert data. Skipping filtering.", flush=True)


    # Avoid duplicated Tier column after merge
    if 'Tier (1/2/3)' in df_ai.columns and 'Tier (1/2/3)' in df_expert_clean.columns:
        df_expert_clean = df_expert_clean.drop(columns=['Tier (1/2/3)'])

    merged_df = pd.merge(df_ai, df_expert_clean, on='Control Reference', how='inner')
    
    print(f"Merged {len(merged_df)} rows (Intersection of AI and Expert data).", flush=True)

    print("Loading Sentence Transformer model (this may take a moment)...", flush=True)
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error initializing SentenceTransformer: {e}", flush=True)
        return

    print("Initializing Google Translator...", flush=True)
    translator = GoogleTranslator(source='auto', target='en')

    print("Calculating similarity metrics and LLM judgments...", flush=True)
    
    # Initialize metrics columns
    merged_df['Semantic_Score'] = 0.0
    merged_df['Comparison_Score'] = 0.0
    merged_df['Expert_Answer_Translated'] = ""
    merged_df['AI_Answer_Translated'] = ""
    merged_df['Validation_Score_LLM_as_judge'] = 0.0
    merged_df['Reasoning_LLM_as_judge'] = ""
    
    llm = get_llm()

    def get_llm_judgment(expert_ans, ai_ans, retries=3):
        prompt = f"""You are an expert auditor evaluating an AI's answer against a Ground Truth expert answer.
Evaluate how accurately the AI answer captures the factual essence of the Ground Truth.
Give a score from 0 to 100, where 100 means the AI fully captures the meaning, and 0 means it completely missed it or contradicted it.

Ground Truth Expert Answer:
{expert_ans}

AI Generated Answer:
{ai_ans}

Provide your response in JSON format with exactly two keys: "score" (an integer from 0 to 100) and "reasoning" (a brief string explaining the score)."""
        for attempt in range(retries):
            try:
                response = llm.invoke(prompt)
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                result = json.loads(content)
                return float(result.get("score", 0)), str(result.get("reasoning", "No reasoning provided."))
            except Exception as e:
                err_str = str(e)
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"Error during LLM judgment after {retries} attempts: {e}", flush=True)
                    return 0.0, f"Error: {e}"
    
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
        
        # 1. Semantic Similarity (Cosine)
        try:
            embeddings = model.encode([ai_ans_en, expert_ans_en])
            cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            # Map [-1, 1] to [0, 100]
            semantic_score = float((cosine_sim + 1) / 2 * 100)
        except Exception as e:
            print(f"Error calculating semantic similarity for row {idx}: {e}")
            semantic_score = 0.0
        
        # 2. LLM AS A JUDGE
        llm_score, llm_reasoning = get_llm_judgment(expert_ans_en, ai_ans_en)
        
        # 3. Final Score
        final_score = semantic_score
        
        merged_df.loc[idx, 'Semantic_Score'] = semantic_score
        merged_df.loc[idx, 'Comparison_Score'] = final_score
        merged_df.loc[idx, 'Expert_Answer_Translated'] = expert_ans_en
        merged_df.loc[idx, 'AI_Answer_Translated'] = ai_ans_en
        merged_df.loc[idx, 'Validation_Score_LLM_as_judge'] = llm_score
        merged_df.loc[idx, 'Reasoning_LLM_as_judge'] = llm_reasoning
        
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
