import json
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from config import CONFIG, PROJECT_ROOT
from rag_engine import RagEngine
from llm_factory import get_llm
from langchain_core.messages import HumanMessage
import os
import time

class RcmAuditor:
    def __init__(self):
        self.rag_engine = RagEngine()
        self.llm = get_llm()
        # Templates are in the project root 'templates' folder
        template_dir = os.path.join(PROJECT_ROOT, 'templates')
        # Create directory if it doesn't exist to avoid errors, though templates should be there
        if not os.path.exists(template_dir):
            print(f"Warning: Template directory not found at {template_dir}")
            # Fallback to local if running from root without package structure? 
            # But PROJECT_ROOT should be correct.
            
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        
    def initialize_rag(self):
        self.rag_engine.build_index()

    def generate_client_summary(self):
        output_file = "outputs/client_summary.txt"
        if os.path.exists(output_file):
            print(f"Client summary already exists at {output_file}")
            return
            
        print("Generating Client Summary...")
        # Retrieve context from client docs acting as a broad query
        query = "Summarize the Provisioning Policy and key credit risk methodologies."
        # Ensure we are querying the client index specifically if needed, but retrieve handles defaults
        docs = self.rag_engine.retrieve(query, k=10) 
        
        context_text = "\n\n".join([d.page_content for d in docs])
        
        prompt = (
            f"You are an expert Auditor. Summarize the following client policy documents representing their "
            f"Provisioning and Credit Risk methodology.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Output a professional executive summary."
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Client summary saved to {output_file}")
        except Exception as e:
            print(f"Error generating client summary: {e}")

    def process_row(self, row):
        # a) Combine 'Control Reference' + 'Design Effectiveness Assessment' (+ optional Test Procedure) from CSV into a query.
        control_ref = row.get('Control Reference', 'Unknown')
        # KEY FIX: Use the question/assessment intent, not just the test steps.
        design_assessment = row.get('Design Effectiveness Assessment', '')
        test_procedure = row.get('Test Procedures', row.get('Test Procedure', ''))
        
        # Construct a richer query
        query = f"Control Ref: {control_ref}. Question: {design_assessment} (Procedure: {test_procedure})"
        
        # b) Retrieve context using the new Spanish-translation logic (handled in RagEngine)
        retrieved_docs = self.rag_engine.retrieve(query, k=10)
        context_text = "\n\n".join([f"[Page {d.metadata.get('page', 'N/A')}] {d.page_content}" for d in retrieved_docs])
        evidence_used = [f"Page {d.metadata.get('page', 'N/A')}" for d in retrieved_docs]

        # c) Call the LLM with a prompt from the template
        template = self.jinja_env.get_template('auditor_response.j2')
        prompt_text = template.render(context=context_text, query=query)

        # Retry logic for generation
        from google.api_core.exceptions import ResourceExhausted

        max_retries = 5
        base_delay = 20 # Start with 20 seconds as requested

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                break
            except Exception as e:
                # Check for ResourceExhausted or similar 429
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt) # Exponential backoff: 20, 40, 80...
                        print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        raise e # Re-raise if retries exhausted
                else:
                    raise e # Re-raise other errors immediately
        
        full_response = response.content
        
        # Parse Verdict
        compliance_verdict = "Insufficient Info"
        if "**COMPLIANCE VERDICT:**" in full_response:
            parts = full_response.split("**COMPLIANCE VERDICT:**")
            generated_answer = parts[0].strip()
            # Clean up verdict
            verdict_lines = parts[1].strip().split('\n')
            for line in verdict_lines:
                clean_line = line.strip().lower()
                if "compliant" in clean_line or "non-compliant" in clean_line or "partial" in clean_line or "insufficient" in clean_line:
                     # Simple extraction
                    if "non-compliant" in clean_line:
                        compliance_verdict = "Non-Compliant"
                    elif "compliant" in clean_line:
                        compliance_verdict = "Compliant"
                    elif "partial" in clean_line:
                        compliance_verdict = "Partial"
                    elif "insufficient" in clean_line:
                        compliance_verdict = "Insufficient Info"
                    break
        else:
            generated_answer = full_response

        # d) VALIDATION STEP: Score the answer (0-10)
        critique_template = self.jinja_env.get_template('auditor_critique.j2')
        validation_prompt = critique_template.render(context=context_text, query=query, answer=generated_answer)

        critique_response = None
        for attempt in range(max_retries):
            try:
                critique_response = self.llm.invoke([HumanMessage(content=validation_prompt)])
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"Rate limit hit during critique. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        print(f"Critique failed after retries: {e}")
                        validation_result = {'score': 0, 'reasoning': f"Rate Limit Error: {e}"}
                        # Skip parsing and return failure result
                        break
                else:
                    print(f"Critique failed with non-rate-limit error: {e}")
                    validation_result = {'score': 0, 'reasoning': f"Error: {e}"}
                    break

        if critique_response:
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
        result['Compliance_Verdict'] = compliance_verdict
        result['Evidence_Sources'] = ", ".join(evidence_used[:5]) # Top 5 pages
        
        return result
