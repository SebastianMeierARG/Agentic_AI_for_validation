import pandas as pd
import json
import os
import time
from config import CONFIG
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import re

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
    cols_to_keep = ['Control Reference', 'Answers based on Clients data']
    df_expert_clean = df_expert[cols_to_keep].dropna(subset=['Answers based on Clients data'])

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

    print("Calculating similarity metrics...", flush=True)
    
    # Initialize metrics columns
    merged_df['Semantic_Score'] = 0.0
    merged_df['Lexical_Score'] = 0.0
    merged_df['Comparison_Score'] = 0.0

    def jaccard_similarity(str1, str2):
        set1 = set(re.findall(r'\w+', str1.lower()))
        set2 = set(re.findall(r'\w+', str2.lower()))
        if not set1 or not set2:
            return 0.0
        return float(len(set1.intersection(set2)) / len(set1.union(set2)))
    
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
        
        # 2. Lexical Similarity (Jaccard)
        jaccard_sim = jaccard_similarity(ai_ans_en, expert_ans_en)
        lexical_score = jaccard_sim * 100
        
        # 3. Final Weighted Score
        # We weigh the semantic meaning highly (80%) but require some factual/term overlap (20%).
        final_score = (semantic_score * 0.8) + (lexical_score * 0.2)
        
        merged_df.at[idx, 'Semantic_Score'] = semantic_score
        merged_df.at[idx, 'Lexical_Score'] = lexical_score
        merged_df.at[idx, 'Comparison_Score'] = final_score
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(merged_df)}...", flush=True)
            merged_df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
            time.sleep(1) # Polite delay
            
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
    print(f"Validation complete. Report saved to {output_path}", flush=True)

if __name__ == "__main__":
    validate_audit()
