import pandas as pd
import json
import os
import time
from config import CONFIG
from llm_factory import get_llm
from langchain_core.messages import HumanMessage

def validate_audit():
    print("Starting Validation Process...")

    # 1. Load AI Results
    output_json = CONFIG['paths']['output_json']
    if not os.path.exists(output_json):
        print(f"Error: {output_json} not found. Run the audit first.")
        return

    print(f"Loading AI results from {output_json}...")
    with open(output_json, 'r', encoding='utf-8') as f:
        ai_results = json.load(f)
    
    df_ai = pd.DataFrame(ai_results)
    
    # Ensure Control Reference is string and clean
    if 'Control Reference' in df_ai.columns:
        df_ai['Control Reference'] = df_ai['Control Reference'].astype(str).str.strip()
    else:
        print("Error: 'Control Reference' column missing in AI results.")
        return

    # 2. Load Expert Answers
    expert_csv = r'c:\Users\semeier\Desktop\gemini_chat_private_GH\Agentic_AI_for_validation\rcm_expert_answer.csv'
    if not os.path.exists(expert_csv):
        print(f"Error: {expert_csv} not found.")
        return

    print(f"Loading Expert answers from {expert_csv}...")
    try:
        # Based on inspection, separator is ';'
        df_expert = pd.read_csv(expert_csv, sep=';', encoding='latin-1') 
    except Exception as e:
        print(f"Error reading expert CSV: {e}")
        return

    # Ensure Control Reference is string and clean
    if 'Control Reference' in df_expert.columns:
        df_expert['Control Reference'] = df_expert['Control Reference'].astype(str).str.strip()
    else:
        print("Error: 'Control Reference' column missing in expert CSV.")
        return

    # 3. Merge DataFrames
    print("Merging data...")
    # Select relevant columns from expert to avoid clutter
    cols_to_keep = ['Control Reference', 'Answers based on Clients data']
    df_expert_clean = df_expert[cols_to_keep].dropna(subset=['Answers based on Clients data'])

    merged_df = pd.merge(df_ai, df_expert_clean, on='Control Reference', how='inner')
    
    print(f"Merged {len(merged_df)} rows (Intersection of AI and Expert data).")

    # 4. LLM Comparison
    import copy
    override_config = copy.deepcopy(CONFIG)
    override_config['llm_settings']['provider'] = 'google'
    override_config['llm_settings']['google'] = {'model': 'models/gemini-flash-latest'}
    
    llm = get_llm(override_config)
    
    comparison_scores = []
    comparison_reasonings = []

    print("Running LLM comparison on rows...", flush=True)
    
    # Create empty column in dataframe if not exists
    merged_df['Comparison_Score'] = 0
    merged_df['Comparison_Reasoning'] = ""
    
    output_path = r'c:\Users\semeier\Desktop\gemini_chat_private_GH\Agentic_AI_for_validation\validation_comparison_report.csv'

    
    for idx, row in merged_df.iterrows():
        question = row.get('Design Effectiveness Assessment', 'N/A')
        ai_ans = row.get('AI_Answer', 'N/A')
        expert_ans = row.get('Answers based on Clients data', 'N/A')
        
        prompt = f"""
You are an expert Audit Supervisor. Compare the "AI Answer" with the "Expert Ground Truth".

Question: {question}

Expert Ground Truth:
{expert_ans}

AI Answer:
{ai_ans}

Task:
1. Rate the semantic similarity and factual accuracy of the AI Answer compared to the Expert Answer on a scale of 0 to 100.
2. Provide a brief reasoning (1 sentence).

Output strictly in JSON format: {{"score": <int>, "reasoning": "<string>"}}
"""
        # Retry logic for rate limits
        max_retries = 3
        score = 0
        reasoning = "Error or Rate Limit"
        
        for attempt in range(max_retries):
            try:
                # Add timeout to avoid hanging?
                response = llm.invoke([HumanMessage(content=prompt)]) 
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]
                
                data = json.loads(content)
                score = data.get('score', 0)
                reasoning = data.get('reasoning', '')
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(2 * (attempt + 1))
                else:
                    reasoning = f"Error: {e}"
                    print(f"Error on row {idx}: {e}", flush=True)
                    break
        
        merged_df.at[idx, 'Comparison_Score'] = score
        merged_df.at[idx, 'Comparison_Reasoning'] = reasoning
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(merged_df)}...", flush=True)
            merged_df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
            time.sleep(1) # Polite delay
            
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
    print(f"Validation complete. Report saved to {output_path}", flush=True)

if __name__ == "__main__":
    validate_audit()
