# Model Performance Validation Report

**Source Data:** `outputs/validation_comparison_report.csv`
**Overall Average Accuracy (LLM):** 69.84 / 100
**Overall Cross-Encoder Similarity:** 56.82 / 100
**Overall Average Stability:** 73.59 / 100
**Overall Average Drift Resistance:** 76.72 / 100
**Overall Average Guardrail Effectiveness:** 91.25 / 100

## 1. Performance by Section (Scope)
This section shows how the AI performed across different thematic areas of the audit.

| Scope | Avg Accuracy | Cross-Encoder Sim. | Avg Stability | Avg Drift | Avg Guardrail | Controls |
|---|---|---|---|---|---|---|
| Forward-Looking Information | 90.00 | 56.96 | 85.00 | 95.00 | 100.00 | 1 |
| Macro Scenarios | 85.00 | 55.84 | 80.00 | 90.00 | 95.00 | 1 |
| PD | 80.00 | 59.20 | 77.50 | 78.75 | 91.25 | 4 |
| Segmentation | 78.33 | 56.34 | 80.00 | 76.67 | 91.67 | 3 |
| ECL Calculation | 77.50 | 51.83 | 77.50 | 85.00 | 92.50 | 4 |
| Model Overrides | 77.50 | 59.06 | 77.50 | 85.00 | 92.50 | 2 |
| LGD | 70.00 | 58.14 | 75.00 | 83.75 | 92.50 | 4 |
| Model Monitoring | 70.00 | 57.69 | 75.00 | 80.00 | 90.00 | 1 |
| Model Governance and Oversight | 65.00 | 60.00 | 72.50 | 77.50 | 91.25 | 4 |
| EAD / CCF | 61.67 | 54.72 | 65.00 | 73.33 | 93.33 | 3 |
| Definition of Default | 60.00 | 58.25 | 67.50 | 60.00 | 85.00 | 2 |
| Risk Contagion | 52.50 | 55.96 | 65.00 | 60.00 | 87.50 | 2 |
| Data Quality and Sources | 20.00 | 51.28 | 50.00 | 30.00 | 80.00 | 1 |

## 2. Strengths: High-Performing Controls (Score >= 80)
These are areas where the AI's answer strongly aligned with the Human Expert's ground truth, successfully capturing the necessary facts and methodological compliance.

### Control 1.1 (Score: 90.0)
- **Scope:** Model Governance and Oversight
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth by confirming the existence of the 'Política de Previsionamiento' and its approval process. However, it introduces additional details about the ECL calculation and governance that, while relevant, slightly diverge from the original focus. The reasoning is mostly consistent and logical, but the introduction of specific points from IFRS 9 may indicate a minor drift. The tone remains professional and objective throughout.

### Control 3.6 (Score: 90.0)
- **Scope:** Segmentation
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth by stating that the segmentation lacks quantitative backing. It provides additional context and examples, which enhances understanding but slightly diverges from the original simplicity. The reasoning is mostly consistent and logical, though it introduces some complexity that could lead to minor drift. The tone remains professional and objective throughout, effectively avoiding inappropriate content.

### Control 7.5 (Score: 90.0)
- **Scope:** LGD
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth by addressing the lack of empirical support for the LGD assumption, aligning well with IFRS 9 requirements. It demonstrates logical consistency and stability in reasoning, although it could be slightly more concise. The answer remains focused on the context without introducing unrelated information, and it maintains a professional tone throughout.

### Control 10.1 (Score: 90.0)
- **Scope:** Forward-Looking Information
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth by stating the lack of a formal process for model selection and validation. It maintains logical consistency and does not introduce irrelevant information, demonstrating strong drift resistance. The tone is professional and objective, adhering to auditing standards.

### Control 6.2 (Score: 85.0)
- **Scope:** PD
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth regarding the PD model's compliance with IFRS 9, but it introduces additional context that may not be explicitly stated in the Ground Truth. The reasoning is mostly consistent, but the introduction of external sources and specific time frames could indicate some drift. The tone remains professional and objective, effectively maintaining guardrails.

### Control 6.3 (Score: 85.0)
- **Scope:** PD
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth regarding the limitations of the bank's PD estimation methodology, though it lacks some specific details about the exclusions mentioned in the expert answer. The reasoning is logically stable and consistent, but there are minor gaps in the depth of analysis. The AI remains focused on the relevant context without introducing unrelated information, and it maintains a professional tone throughout.

### Control 5.3 (Score: 85.0)
- **Scope:** Risk Contagion
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth regarding the lack of controls and periodic reviews for the contagion methodology, but it slightly misrepresents the specifics of the Chain Ladder technique. The reasoning is consistent and logically stable, with a strong focus on the context provided. It remains grounded without introducing external information, and maintains a professional tone throughout.

### Control 11.2 (Score: 85.0)
- **Scope:** ECL Calculation
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth by addressing the lack of formal criteria for staging under IFRS 9, but it introduces additional details about ECL calculations that, while relevant, may not be explicitly stated in the Ground Truth. The reasoning is logically consistent and stable, though it could be seen as slightly more detailed than necessary. The answer remains focused on the context without introducing unrelated information, demonstrating strong drift resistance. The tone is professional and objective, effectively maintaining guardrails.

### Control 11.1 (Score: 85.0)
- **Scope:** ECL Calculation
- **Why it did well:** The AI answer accurately identifies issues with the integration of quantitative and qualitative factors in the ECL calculation, aligning well with the Ground Truth. However, it introduces additional details about PD, LGD, EAD, and FLI that, while relevant, may not be explicitly mentioned in the Ground Truth, affecting its accuracy slightly. The reasoning is consistent and logically structured, but the introduction of specific page references could lead to minor inconsistencies. The answer remains focused on the context without drifting into unrelated information, and it maintains a professional tone throughout.

### Control 9.3 (Score: 85.0)
- **Scope:** Macro Scenarios
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth by addressing the lack of multiple, probability-weighted scenarios in the methodology, which aligns with IFRS 9 requirements. However, it introduces some additional context about the bank's commitment to updating documentation that is not explicitly mentioned in the Ground Truth, which slightly affects accuracy. The reasoning is logically stable and consistent, but the introduction of new information could lead to minor drift. The tone remains professional and objective throughout.

### Control 8.1 (Score: 85.0)
- **Scope:** EAD / CCF
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth by addressing the lack of a comprehensive methodology for EAD calculation, particularly for off-balance sheet items. However, it slightly misrepresents the specific treatment of EAD when debt exceeds limits, which affects accuracy. The reasoning is logically stable and consistent, but there are minor gaps in the depth of analysis regarding empirical support for CCF. The AI remains focused on the context without introducing unrelated information, demonstrating strong drift resistance. The tone is professional and objective, effectively maintaining guardrails.

### Control 13.1 (Score: 85.0)
- **Scope:** Model Overrides
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth by stating that there is no formal policy regarding model overrides, but it lacks some specificity about the governance implications mentioned in the expert answer. The reasoning is consistent and logically stable, though it could be more concise. The AI remains focused on the context without introducing unrelated information, demonstrating strong drift resistance. The tone is professional and objective, effectively maintaining guardrails.

### Control 4.2 (Score: 80.0)
- **Scope:** Definition of Default
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth regarding the methodology change, but it introduces additional details that may not be directly relevant, affecting drift resistance. The reasoning is mostly consistent but could be clearer in linking the points. The professional tone is maintained throughout.

### Control 6.5 (Score: 80.0)
- **Scope:** PD
- **Why it did well:** The AI answer accurately identifies the lack of formal analyses and validations as highlighted in the Ground Truth, but it introduces some extraneous details about future plans and timelines that were not present in the original context. The reasoning is mostly consistent but could be seen as slightly speculative regarding future actions. The AI maintains a professional tone, but the introduction of future commitments could be seen as drifting from the immediate context.

### Control 8.6 (Score: 80.0)
- **Scope:** EAD / CCF
- **Why it did well:** The AI answer accurately captures the essence of the Ground Truth by stating that no specific controls for multi-currency facilities are mentioned. However, it slightly misinterprets the context by implying a lack of documentation rather than directly addressing the absence of identified exposures. The reasoning is consistent and logically stable, with no significant drift from the topic. The tone remains professional and objective throughout.

## 3. Weaknesses: Low-Performing Controls (Score <= 40)
These are areas where the AI struggled to match the Human Expert, failing to capture specific nuances, missing quantitative validations, or misunderstanding the methodology requirements versus simple documentation.

### Control 2.1 (Score: 20.0)
- **Scope:** Data Quality and Sources
- **Why it failed (Reasoning):** The AI answer diverges significantly from the Ground Truth, which discusses accounting policies and documentation rather than the validation of an ECL model. The AI's focus on IFRS 9 and credit loss estimation is unrelated to the original context, indicating a lack of accuracy and drift resistance. Stability is moderate as the reasoning is internally consistent, but it is not relevant to the provided context. The tone is professional, contributing to a higher guardrail score.

### Control 5.1 (Score: 20.0)
- **Scope:** Risk Contagion
- **Why it failed (Reasoning):** The AI answer diverges significantly from the Ground Truth, focusing on risk contagion rather than the Point-in-Time methodology and the historical window adjustment mentioned. While it maintains a professional tone, the lack of relevance to the core topic and the introduction of unrelated concepts indicate poor accuracy and drift resistance.

### Control 8.2 (Score: 20.0)
- **Scope:** EAD / CCF
- **Why it failed (Reasoning):** The AI answer inaccurately states that the bank has defined CCFs for off-balance sheet exposures, contradicting the Ground Truth which indicates the absence of a formal model. The reasoning lacks consistency and does not align with the factual essence of the expert answer. However, the tone remains professional and objective.

### Control 1.3 (Score: 40.0)
- **Scope:** Model Governance and Oversight
- **Why it failed (Reasoning):** The AI answer diverges significantly from the Ground Truth, which confirms the existence of detailed methodological chapters in the policy. While the AI provides relevant information about the lack of independent validation, it fails to address the specific chapters mentioned in the Ground Truth, leading to a low accuracy score. The reasoning is somewhat stable but lacks direct relevance to the original question, resulting in a moderate stability score. The AI remains focused on the context without introducing unrelated information, hence a decent drift resistance score. The tone is professional and objective, maintaining a high guardrail effectiveness.

### Control 4.1 (Score: 40.0)
- **Scope:** Definition of Default
- **Why it failed (Reasoning):** The AI answer diverges significantly from the Ground Truth, focusing on the definition of default rather than the specific methodology of 'contagio de stage' and the absence of UTP criteria for Stage 3. While it provides some relevant information, it does not accurately capture the essence of the Ground Truth. The reasoning is somewhat stable but lacks direct relevance to the core topic. The answer does not hallucinate external information but does drift from the main context. The tone is professional and objective, maintaining guardrails effectively.

## 4. Conclusions and Recommendations
1. **Analyze the lowest performing Scopes** to identify if the AI requires better prompts regarding specific mathematical concepts (e.g., LGD factors or PD validation).
2. **Evaluate the reasoning in Section 3** to see if the AI is being too literal or if the human expert's ground truth relies on out-of-context knowledge not present in the RAG documents.
3. **Consider adding targeted few-shot examples** to the prompt for the worst-performing 'Control Reference' types if they follow a discernible pattern.

## 5. Advanced Validation Metrics Interpretation
The following metrics explain the AI's performance and generation quality:

- **Accuracy (LLM Judge) (69.84 / 100):** How well the AI's answer captures the factual essence and methodologies of the human expert's Ground Truth.
- **Cross-Encoder Similarity (56.82 / 100):** A direct neural-network comparison of the AI output vs. the Expert Reference. High scores indicate strong semantic alignment and structural matching, capturing logical entailment better than simple keyword cosine-similarity.
- **Stability (73.59 / 100):** Measures the logical consistency and robustness of the AI's reasoning. A score below 80 suggests the AI's rationale for its audit findings is fragile or logically flawed even if factually adjacent.
- **Drift Resistance (76.72 / 100):** Evaluates how well the AI stays grounded in the context provided. A low score indicates the AI is hallucinating external information or straying from the specific methodology.
- **Guardrail Effectiveness (91.25 / 100):** Checks for appropriate professional auditing tone and avoidance of prohibited content. High scores confirm the AI maintains objective, safe, and professional outputs.
