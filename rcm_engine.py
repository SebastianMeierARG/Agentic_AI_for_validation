import json
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import CONFIG
from rag_engine import RagEngine
import os

class RcmAuditor:
    def __init__(self):
        self.rag_engine = RagEngine()
        self.llm = ChatOpenAI(
            model=CONFIG['llm_settings']['model'],
            temperature=CONFIG['llm_settings']['temperature']
        )
        # Assuming templates are in the 'templates' directory relative to this file
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        # Create directory if it doesn't exist to avoid errors, though templates should be there
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
            
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        
        # We will use inline prompts if templates are missing, or load them if they exist.
        # For robustness, defining simple templates here as fallback or primary if files aren't populated.
        # But per existing structure, we expect files. I will assume we can use string formatting 
        # or just create the prompts in code to ensure they match the instructions perfectly.
        
    def initialize_rag(self):
        self.rag_engine.build_index()

    def process_row(self, row):
        # a) Combine 'Control Reference' + 'Test Procedure' from CSV into a query.
        control_ref = row.get('Control Reference', 'Unknown')
        test_procedure = row.get('Test Procedures', row.get('Test Procedure', ''))
        
        query = f"Control Ref: {control_ref}. Procedure: {test_procedure}"
        
        # b) Retrieve context using the new Spanish-translation logic (handled in RagEngine)
        retrieved_docs = self.rag_engine.retrieve(query, k=10)
        context_text = "\n\n".join([f"[Page {d.metadata.get('page', 'N/A')}] {d.page_content}" for d in retrieved_docs])
        evidence_used = [f"Page {d.metadata.get('page', 'N/A')}" for d in retrieved_docs]

        # c) Call the LLM with a prompt that forces it to answer strictly based on context
        prompt_text = f"""
You are an expert IFRS9 Auditor. Answer the audit test procedure strictly based *ONLY* on the provided Context.
If the context does not contain the answer, state "Not Documented in provided context".
You must cite the Page number for every assertion.

Context:
{context_text}

Audit Query:
{query}

Answer (Strictly evidence-based, cite pages):
"""
        response = self.llm.invoke([HumanMessage(content=prompt_text)])
        generated_answer = response.content

        # d) VALIDATION STEP: Score the answer (0-10)
        validation_prompt = f"""
You are a Lead Auditor Quality/Assurance reviewer. 
Review the following "AI Generated Answer" against the provided "Context" and the "Audit Query".
Score the answer from 0 to 10 on how faithfully it handles the evidence.
0 = Hallucination or irrelevant.
10 = Perfect citations and strict adherence to context.

Return ONLY a JSON object with keys: "score" (int), "reasoning" (string).

Context:
{context_text}

Audit Query:
{query}

AI Generated Answer:
{generated_answer}
"""
        critique_response = self.llm.invoke([HumanMessage(content=validation_prompt)])
        try:
            content = critique_response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            validation_result = json.loads(content)
        except Exception as e:
            print(f"Error parsing validation JSON: {e}")
            validation_result = {'score': 0, 'reasoning': f"Parse Error: {e}"}

        # Construct result
        result = row.copy()
        result['AI_Answer'] = generated_answer
        result['Validation_Score'] = validation_result.get('score', 0)
        result['Validation_Reasoning'] = validation_result.get('reasoning', '')
        result['Evidence_Sources'] = ", ".join(evidence_used[:5]) # Top 5 pages
        
        return result
