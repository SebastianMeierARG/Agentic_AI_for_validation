import pandas as pd
import argparse
import os

def generate_report(input_csv, output_md):
    """Generates a Markdown report analyzing AI performance from the validation CSV."""
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found.")
        return

    # Load data
    df = pd.read_csv(input_csv, sep=';')
    
    # Ensure necessary columns exist
    required_cols = ['Scope', 'Control Reference', 'Validation_Score_LLM_as_judge', 'Reasoning_LLM_as_judge', 'Cross_Encoder_Score', 'Stability_Score', 'Drift_Resistance_Score', 'Guardrail_Effectiveness_Score']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' missing from CSV.")
            return

    # Calculate overall metrics
    avg_score = df['Validation_Score_LLM_as_judge'].mean()
    avg_cross_encoder_score = df['Cross_Encoder_Score'].mean()
    avg_stability = df['Stability_Score'].mean()
    avg_drift = df['Drift_Resistance_Score'].mean()
    avg_guardrail = df['Guardrail_Effectiveness_Score'].mean()
    
    # Calculate metrics by Scope
    scope_metrics = df.groupby('Scope').agg({
        'Validation_Score_LLM_as_judge': ['mean', 'count'],
        'Cross_Encoder_Score': 'mean',
        'Stability_Score': 'mean',
        'Drift_Resistance_Score': 'mean',
        'Guardrail_Effectiveness_Score': 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    scope_metrics.columns = ['Scope', 'Avg_LLM_Score', 'Count', 'Avg_Cross_Encoder', 'Avg_Stability', 'Avg_Drift', 'Avg_Guardrail']
    scope_metrics = scope_metrics.sort_values(by='Avg_LLM_Score', ascending=False)

    # Identify High and Low performers (based on LLM-as-a-judge score)
    high_performers = df[df['Validation_Score_LLM_as_judge'] >= 80].sort_values(by='Validation_Score_LLM_as_judge', ascending=False)
    low_performers = df[df['Validation_Score_LLM_as_judge'] <= 40].sort_values(by='Validation_Score_LLM_as_judge', ascending=True)

    # Begin writing the Markdown report
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Model Performance Validation Report\n\n")
        f.write(f"**Source Data:** `{input_csv}`\n")
        f.write(f"**Overall Average Accuracy (LLM):** {avg_score:.2f} / 100\n")
        f.write(f"**Overall Cross-Encoder Similarity:** {avg_cross_encoder_score:.2f} / 100\n")
        f.write(f"**Overall Average Stability:** {avg_stability:.2f} / 100\n")
        f.write(f"**Overall Average Drift Resistance:** {avg_drift:.2f} / 100\n")
        f.write(f"**Overall Average Guardrail Effectiveness:** {avg_guardrail:.2f} / 100\n\n")
        
        f.write("## 1. Performance by Section (Scope)\n")
        f.write("This section shows how the AI performed across different thematic areas of the audit.\n\n")
        f.write("| Scope | Avg Accuracy | Cross-Encoder Sim. | Avg Stability | Avg Drift | Avg Guardrail | Controls |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, row in scope_metrics.iterrows():
            f.write(f"| {row['Scope']} | {row['Avg_LLM_Score']:.2f} | {row['Avg_Cross_Encoder']:.2f} | {row['Avg_Stability']:.2f} | {row['Avg_Drift']:.2f} | {row['Avg_Guardrail']:.2f} | {row['Count']} |\n")
        f.write("\n")

        f.write("## 2. Strengths: High-Performing Controls (Score >= 80)\n")
        f.write("These are areas where the AI's answer strongly aligned with the Human Expert's ground truth, successfully capturing the necessary facts and methodological compliance.\n\n")
        if high_performers.empty:
            f.write("*No controls scored 80 or above.*\n\n")
        else:
            for _, row in high_performers.iterrows():
                f.write(f"### Control {row['Control Reference']} (Score: {row['Validation_Score_LLM_as_judge']})\n")
                f.write(f"- **Scope:** {row['Scope']}\n")
                f.write(f"- **Why it did well:** {row['Reasoning_LLM_as_judge']}\n\n")

        f.write("## 3. Weaknesses: Low-Performing Controls (Score <= 40)\n")
        f.write("These are areas where the AI struggled to match the Human Expert, failing to capture specific nuances, missing quantitative validations, or misunderstanding the methodology requirements versus simple documentation.\n\n")
        if low_performers.empty:
            f.write("*No controls scored 40 or below.*\n\n")
        else:
            for _, row in low_performers.iterrows():
                f.write(f"### Control {row['Control Reference']} (Score: {row['Validation_Score_LLM_as_judge']})\n")
                f.write(f"- **Scope:** {row['Scope']}\n")
                f.write(f"- **Why it failed (Reasoning):** {row['Reasoning_LLM_as_judge']}\n\n")

        f.write("## 4. Conclusions and Recommendations\n")
        f.write("1. **Analyze the lowest performing Scopes** to identify if the AI requires better prompts regarding specific mathematical concepts (e.g., LGD factors or PD validation).\n")
        f.write("2. **Evaluate the reasoning in Section 3** to see if the AI is being too literal or if the human expert's ground truth relies on out-of-context knowledge not present in the RAG documents.\n")
        f.write("3. **Consider adding targeted few-shot examples** to the prompt for the worst-performing 'Control Reference' types if they follow a discernible pattern.\n\n")

        f.write("## 5. Advanced Validation Metrics Interpretation\n")
        f.write("The following metrics explain the AI's performance and generation quality:\n\n")
        f.write(f"- **Accuracy (LLM Judge) ({avg_score:.2f} / 100):** How well the AI's answer captures the factual essence and methodologies of the human expert's Ground Truth.\n")
        f.write(f"- **Cross-Encoder Similarity ({avg_cross_encoder_score:.2f} / 100):** A direct neural-network comparison of the AI output vs. the Expert Reference. High scores indicate strong semantic alignment and structural matching, capturing logical entailment better than simple keyword cosine-similarity.\n")
        f.write(f"- **Stability ({avg_stability:.2f} / 100):** Measures the logical consistency and robustness of the AI's reasoning. A score below 80 suggests the AI's rationale for its audit findings is fragile or logically flawed even if factually adjacent.\n")
        f.write(f"- **Drift Resistance ({avg_drift:.2f} / 100):** Evaluates how well the AI stays grounded in the context provided. A low score indicates the AI is hallucinating external information or straying from the specific methodology.\n")
        f.write(f"- **Guardrail Effectiveness ({avg_guardrail:.2f} / 100):** Checks for appropriate professional auditing tone and avoidance of prohibited content. High scores confirm the AI maintains objective, safe, and professional outputs.\n")

    print(f"Report successfully generated at: {output_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a markdown summary report from the validation CSV.")
    parser.add_argument("--input", "-i", type=str, default="outputs/validation_comparison_report.csv", help="Path to the validation comparison CSV.")
    parser.add_argument("--output", "-o", type=str, default="outputs/model_performance_report.md", help="Path to save the generated markdown report.")
    
    args = parser.parse_args()
    generate_report(args.input, args.output)
