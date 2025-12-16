import os
import json
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import CONFIG
from rag_engine import RagEngine

class RcmAuditor:
    def __init__(self):
        self.rag_engine = RagEngine()
        self.llm = ChatOpenAI(
            model=CONFIG['llm_settings']['model'],
            temperature=CONFIG['llm_settings']['temperature']
        )

        # Setup Jinja2 environment
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))

        # Load templates
        self.response_template = self.jinja_env.get_template('auditor_response.j2')
        self.critique_template = self.jinja_env.get_template('auditor_critique.j2')

    def initialize_rag(self):
        """Explicitly build the RAG index."""
        self.rag_engine.build_index()

    def process_row(self, row):
        """
        Process a single row from the RCM CSV.
        Expected row keys: 'Control Reference', 'Test Procedure' (or similar).
        """
        # Extract query
        # Using strict column names or fallback
        control_ref = row.get('Control Reference', 'Unknown Ref')
        test_procedure = row.get('Test Procedure', '')

        query = f"{control_ref}: {test_procedure}"

        # Retrieve
        retrieved_docs = self.rag_engine.retrieve(query, k=5)
        context_text = "\n\n".join([d.page_content for d in retrieved_docs])
        evidence_used = [f"Page {d.metadata.get('page', 'N/A')}: {d.page_content[:100]}..." for d in retrieved_docs]

        # Generate Answer
        prompt_content = self.response_template.render(context=context_text, question=query)
        response = self.llm.invoke([HumanMessage(content=prompt_content)])
        generated_answer = response.content

        # Validate (Self-Critique)
        validation_result = {}
        if CONFIG['validation']['enable_self_critique']:
            critique_prompt = self.critique_template.render(context=context_text, answer=generated_answer)
            critique_response = self.llm.invoke([HumanMessage(content=critique_prompt)])
            try:
                # Basic cleanup to handle markdown json blocks if model adds them
                content = critique_response.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()

                validation_result = json.loads(content)
            except Exception as e:
                print(f"Error parsing validation JSON: {e}")
                validation_result = {'error': 'Failed to parse JSON', 'raw_output': critique_response.content}

        # Return result dictionary
        result = row.copy()
        result['AI_Answer'] = generated_answer
        result['Validation_Score'] = validation_result.get('score', 'N/A')
        result['Hallucination_Flag'] = validation_result.get('hallucination', 'N/A')
        result['Validation_Reasoning'] = validation_result.get('reasoning', 'N/A')
        result['Evidence_Used'] = evidence_used

        return result

    def run_audit(self, input_csv_path=None, output_json_path=None):
        input_path = input_csv_path or CONFIG['paths']['input_csv']
        output_path = output_json_path or CONFIG['paths']['output_json']

        if not os.path.exists(input_path):
            print(f"Input CSV not found: {input_path}")
            return

        df = pd.read_csv(input_path, sep=';')
        results = []

        # Ensure RAG is ready
        self.initialize_rag()

        print(f"Processing {len(df)} rows...")
        for idx, row in df.iterrows():
            print(f"Processing row {idx + 1}...")
            row_dict = row.to_dict()
            res = self.process_row(row_dict)
            results.append(res)

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Audit complete. Results saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    auditor = RcmAuditor()
    auditor.run_audit()
