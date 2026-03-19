import json
import re
import os
import time
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from config import CONFIG, PROJECT_ROOT
from rag_engine import RagEngine
from llm_factory import get_llm
from langchain_core.messages import HumanMessage

# Patterns that may indicate prompt injection attempts embedded in document content.
_INJECTION_PATTERNS = [
    r'(?im)^(INSTRUCTION|SYSTEM PROMPT|IGNORE (ALL )?(PREVIOUS|PRIOR|ABOVE) INSTRUCTIONS)',
    r'(?i)ignore (all )?(previous|prior|above) instructions',
    r'(?i)you are (now )?a[n]? ',
    r'(?i)(jailbreak|DAN mode|act as if)',
    r'(?im)^NEW TASK[:\s]',
    r'(?im)^FORGET (EVERYTHING|ALL)',
]


def _sanitize_chunk(text: str) -> str:
    """Remove or redact potential prompt-injection patterns from retrieved document chunks."""
    for pattern in _INJECTION_PATTERNS:
        text = re.sub(pattern, '[REDACTED]', text)
    return text


class RcmAuditor:
    def __init__(self):
        self.rag_engine = RagEngine()
        self.llm = get_llm()
        template_dir = os.path.join(PROJECT_ROOT, 'templates')
        if not os.path.exists(template_dir):
            print(f"Warning: Template directory not found at {template_dir}")
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))

        # Lazy-loaded secondary LLM for cross-validation
        self._secondary_llm = None
        self._secondary_llm_loaded = False

    def _get_secondary_llm(self):
        """Return the judge LLM for cross-hallucination checks, loading it once."""
        if not self._secondary_llm_loaded:
            self._secondary_llm_loaded = True
            enabled = CONFIG.get('validation', {}).get('enable_cross_llm_critique', True)
            if not enabled:
                return None
            try:
                from llm_factory import get_judge_llm
                self._secondary_llm = get_judge_llm()
            except Exception as e:
                print(f"Warning: Could not load judge LLM for cross-critique: {e}")
        return self._secondary_llm

    def _invoke_with_retry(self, llm, messages, max_retries=5, base_delay=20, label="LLM"):
        """Invoke an LLM with exponential backoff on rate-limit errors."""
        for attempt in range(max_retries):
            try:
                return llm.invoke(messages)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        print(
                            f"Rate limit hit ({label}). Waiting {wait_time}s "
                            f"(retry {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

    def initialize_rag(self):
        self.rag_engine.build_index()

    def generate_client_summary(self):
        output_file = os.path.join(PROJECT_ROOT, "outputs", "client_summary.md")
        if os.path.exists(output_file):
            print(f"Client summary already exists at {output_file}")
            return

        print("Generating Client Summary...")
        query = (
            "Summarize the client's policy on: Model Governance, Data Quality, Segmentation, "
            "Definition of Default, Risk Contagion, PD, LGD, EAD/CCF, Macro Scenarios, "
            "Forward-Looking Information, ECL Calculation, Model Monitoring, and Model Overrides."
        )

        docs = self.rag_engine.retrieve(query, k=15)
        context_text = "\n\n".join([d.page_content for d in docs])

        try:
            template = self.jinja_env.get_template('client_summary.j2')
            prompt = template.render(context=context_text)
            response = self._invoke_with_retry(self.llm, [HumanMessage(content=prompt)], label="summary")
            summary = response.content
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Client summary saved to {output_file}")
        except Exception as e:
            print(f"Error generating client summary: {e}")

    def _cross_llm_critique(self, context_text: str, answer: str) -> dict:
        """
        Ask an independent LLM to check whether the answer contains claims not
        supported by the context.

        Uses a single provider queue shared across all rows. On any provider
        failure (quota, credit limit, auth, or repeated per-minute rate limits)
        the queue advances to the next provider — no waiting on a broken provider.

        Returns a dict with keys: hallucinated, unsupported_claims, confidence_in_answer.
        Returns None if all providers fail.
        """
        # Build the provider queue once per auditor instance.
        if not hasattr(self, '_cross_llm_queue'):
            from llm_factory import get_judge_llm, get_fallback_judge_llm, get_secondary_llm, get_llm as _get_primary
            self._cross_llm_queue = [get_judge_llm, get_fallback_judge_llm, get_secondary_llm, _get_primary]
            self._cross_llm_idx   = 0
            self._cross_llm_llm   = None  # active provider instance

        prompt = (
            "You are an independent AI auditor reviewing an answer produced by another AI system.\n\n"
            "Your task: Determine whether the answer contains any factual claims that are NOT supported "
            "by the provided context. Do NOT penalise for missing information — only flag invented facts.\n\n"
            "Context (retrieved from official bank documents):\n"
            f"{context_text[:4000]}\n\n"
            "AI-Generated Answer:\n"
            f"{answer}\n\n"
            "Return ONLY valid JSON with these exact keys:\n"
            '{"hallucinated": <true/false>, '
            '"unsupported_claims": ["claim1", "claim2"], '
            '"confidence_in_answer": <integer 0-100>}'
        )

        # Try current provider; on any failure advance the queue.
        while self._cross_llm_idx < len(self._cross_llm_queue):
            # Initialise provider if not yet done (or if we just advanced).
            if self._cross_llm_llm is None:
                fn = self._cross_llm_queue[self._cross_llm_idx]
                try:
                    self._cross_llm_llm = fn()
                except Exception as init_err:
                    print(f"Cross-LLM provider init failed: {init_err}")
                    self._cross_llm_llm = None
                if self._cross_llm_llm is None:
                    self._cross_llm_idx += 1
                    continue

            try:
                # One retry for per-minute rate limits; give up quickly and advance provider.
                response = self._invoke_with_retry(
                    self._cross_llm_llm, [HumanMessage(content=prompt)],
                    max_retries=2, base_delay=5, label="cross-LLM critique"
                )
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:].rstrip("```").strip()
                elif content.startswith("```"):
                    content = content[3:].rstrip("```").strip()
                result = json.loads(content)
                return {
                    "hallucinated":       bool(result.get("hallucinated", False)),
                    "unsupported_claims": result.get("unsupported_claims", []),
                    "confidence_in_answer": int(result.get("confidence_in_answer", 50)),
                }
            except Exception as e:
                print(f"Cross-LLM provider failed ({str(e)[:100]}). Advancing to next provider...")
                self._cross_llm_llm = None   # force re-init of next provider
                self._cross_llm_idx += 1

        print("Cross-LLM critique skipped: all providers exhausted for this run.")
        return None

    @staticmethod
    def _is_provider_exhausted(exc: Exception) -> bool:
        """Return True for errors where retrying the same provider won't help."""
        msg = str(exc)
        if "402" in msg or "credit_limit" in msg.lower():
            return True
        if "401" in msg or "invalid_api_key" in msg.lower() or "invalid api key" in msg.lower():
            return True
        if ("429" in msg or "rate_limit_exceeded" in msg) and "tokens per day" in msg.lower():
            return True
        return False

    def process_row(self, row):
        control_ref = row.get('Control Reference', 'Unknown')
        design_assessment = row.get(
            'Design Effectiveness Assessment 2.0',
            row.get('Design Effectiveness Assessment', '')
        )
        test_procedure = row.get(
            'Test Procedures 2.0',
            row.get('Test Procedures', row.get('Test Procedure', ''))
        )

        query = (
            f"We are auditing '{control_ref}'. The requirement is: '{design_assessment}'. "
            f"Specifically, we must verify the following Test Procedure: '{test_procedure}'."
        )

        retrieved_docs = self.rag_engine.retrieve(query, k=10)

        context_parts = []
        evidence_dict = {}

        for d in retrieved_docs:
            page = d.metadata.get('page', 'N/A')
            source_path = d.metadata.get('source', '')
            filename = os.path.basename(source_path) if source_path else 'Unknown_Document'

            # Sanitize chunk before injecting into prompt
            safe_content = _sanitize_chunk(d.page_content)
            context_parts.append(f"[Page {page} of '{filename}'] {safe_content}")

            if filename not in evidence_dict:
                evidence_dict[filename] = set()
            if page != 'N/A':
                evidence_dict[filename].add(str(page))

        context_text = "\n\n".join(context_parts)

        # Build Evidence Sources string
        evidence_sources_list = []
        for fname, pages in evidence_dict.items():
            if not pages:
                evidence_sources_list.append(f"'{fname}'")
            else:
                try:
                    sorted_pages = sorted(list(pages), key=int)
                except ValueError:
                    sorted_pages = sorted(list(pages))

                if len(sorted_pages) == 1:
                    evidence_sources_list.append(f"Page {sorted_pages[0]} of '{fname}'")
                elif len(sorted_pages) == 2:
                    evidence_sources_list.append(
                        f"Pages {sorted_pages[0]} and {sorted_pages[1]} of '{fname}'"
                    )
                else:
                    pages_str = ", ".join(sorted_pages[:-1]) + f" and {sorted_pages[-1]}"
                    evidence_sources_list.append(f"Pages {pages_str} of '{fname}'")

        evidence_sources_str = "; ".join(evidence_sources_list)

        # --- Generate answer ---
        template = self.jinja_env.get_template('auditor_response.j2')
        prompt_text = template.render(
            context=context_text,
            design_assessment=design_assessment,
            test_procedure=test_procedure,
        )

        response = self._invoke_with_retry(
            self.llm, [HumanMessage(content=prompt_text)], label="answer generation"
        )
        full_response = response.content

        # Parse answer, evidence, and verdict
        answer_match = re.search(
            r'<answer>(.*?)(?:</answer>|<evidence_sources>|\*\*COMPLIANCE|$)',
            full_response, re.DOTALL | re.IGNORECASE
        )
        final_answer = answer_match.group(1).strip() if answer_match else ""

        evidence_match = re.search(
            r'<evidence_sources>(.*?)(?:</evidence_sources>|\*\*COMPLIANCE|$)',
            full_response, re.DOTALL | re.IGNORECASE
        )
        extracted_evidence = evidence_match.group(1).strip() if evidence_match else "None"

        compliance_verdict = "Insufficient Info"
        if "**COMPLIANCE VERDICT:**" in full_response:
            parts = full_response.split("**COMPLIANCE VERDICT:**")
            if not final_answer:
                final_answer = parts[0].strip()
            for line in parts[1].strip().split('\n'):
                clean_line = line.strip().lower()
                if "non-compliant" in clean_line:
                    compliance_verdict = "Non-Compliant"
                    break
                elif "compliant" in clean_line:
                    compliance_verdict = "Compliant"
                    break
                elif "partial" in clean_line:
                    compliance_verdict = "Partial"
                    break
                elif "insufficient" in clean_line:
                    compliance_verdict = "Insufficient Info"
                    break
        else:
            if not final_answer:
                final_answer = full_response

        # --- Self-critique (intrinsic validation) ---
        validation_result = {'score': 0, 'reasoning': 'Not run', 'hallucination_rate': 0.0}
        if CONFIG.get('validation', {}).get('enable_self_critique', True):
            critique_template = self.jinja_env.get_template('auditor_critique.j2')
            validation_prompt = critique_template.render(
                context=context_text, query=query, answer=final_answer
            )
            try:
                critique_response = self._invoke_with_retry(
                    self.llm, [HumanMessage(content=validation_prompt)],
                    label="self-critique"
                )
                content = critique_response.content.strip()
                if content.startswith("```json"):
                    content = content[7:].rstrip("```").strip()
                elif content.startswith("```"):
                    content = content[3:].rstrip("```").strip()
                validation_result = json.loads(content)
            except Exception as e:
                print(f"Self-critique failed: {e}")
                validation_result = {'score': 0, 'reasoning': f"Error: {e}", 'hallucination_rate': 0.0}

        critique_score = validation_result.get('score', 0)

        # --- Cross-LLM hallucination critique ---
        cross_result = self._cross_llm_critique(context_text, final_answer)

        # --- Confidence score (0-100) ---
        # Blend self-critique score (normalised to 0-100) with cross-LLM confidence.
        self_confidence = (critique_score / 10.0) * 100.0
        if cross_result is not None:
            cross_confidence = float(cross_result.get('confidence_in_answer', 50))
            if cross_result.get('hallucinated', False):
                cross_confidence = max(0.0, cross_confidence - 30.0)
            confidence_score = round((self_confidence + cross_confidence) / 2.0, 1)
        else:
            confidence_score = round(self_confidence, 1)

        # --- Build result ---
        result = row.copy()
        result['AI_Answer'] = final_answer
        result['Evidence_Sources'] = extracted_evidence
        result['Verification_Step'] = extracted_evidence
        result['Compliance_Verdict'] = compliance_verdict
        result['Validation_Score'] = critique_score
        result['Validation_Reasoning'] = validation_result.get('reasoning', '')
        result['Hallucination_Rate'] = validation_result.get('hallucination_rate', 0.0)
        result['Confidence_Score'] = confidence_score
        result['Cross_LLM_Hallucinated'] = (
            cross_result.get('hallucinated', False) if cross_result else None
        )
        result['Cross_LLM_Concerns'] = (
            "; ".join(cross_result.get('unsupported_claims', [])) if cross_result else ""
        )

        return result
