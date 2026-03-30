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

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _build_context(self, retrieved_docs):
        """
        Convert retrieved LangChain documents into a sanitized context string
        and an evidence dict keyed by filename → set of page numbers.
        """
        context_parts = []
        evidence_dict = {}

        for d in retrieved_docs:
            page = d.metadata.get('page', 'N/A')
            source_path = d.metadata.get('source', '')
            filename = os.path.basename(source_path) if source_path else 'Unknown_Document'

            safe_content = _sanitize_chunk(d.page_content)
            context_parts.append(f"[Page {page} of '{filename}'] {safe_content}")

            if filename not in evidence_dict:
                evidence_dict[filename] = set()
            if page != 'N/A':
                evidence_dict[filename].add(str(page))

        return "\n\n".join(context_parts), evidence_dict

    def _parse_response(self, full_response):
        """
        Parse the LLM response into (answer, evidence_sources, compliance_verdict, classification).

        Expects a JSON object produced by auditor_response.j2 or auditor_revision.j2.
        Falls back to the original regex approach if JSON parsing fails, so a
        badly-formatted response never silently discards the answer.
        """
        content = full_response.strip()
        if content.startswith("```json"):
            content = content[7:].rstrip("` \n").strip()
        elif content.startswith("```"):
            content = content[3:].rstrip("` \n").strip()

        try:
            data = json.loads(content)
            return (
                data.get("answer", "").strip(),
                data.get("evidence_sources", "None").strip(),
                data.get("compliance_verdict", "Insufficient Info").strip(),
                data.get("classification", "").strip(),
            )
        except (json.JSONDecodeError, ValueError):
            print("Warning: LLM response was not valid JSON — falling back to regex parsing.")

        # --- Regex fallback (original logic) ---
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

        class_match = re.search(
            r'\[CLASSIFICATION:\s*(DOCUMENTATION_CHECK|METHODOLOGY_CHECK|QUANTITATIVE_CHECK)\]',
            full_response, re.IGNORECASE
        )
        classification = class_match.group(1) if class_match else ""

        return final_answer, extracted_evidence, compliance_verdict, classification

    def _verify_citations(self, answer: str, context_text: str) -> dict:
        """
        Check that every [Page X of 'Filename'] citation in the answer actually
        exists in the retrieved context. Returns verification stats.

        A citation is considered verified if the exact marker string
        '[Page X of 'Filename']' appears in context_text — meaning the page
        was genuinely retrieved and not invented.
        """
        citations = re.findall(r"\[Page ([^\]]+) of '([^']+)'\]", answer)
        total = len(citations)
        if total == 0:
            return {"total": 0, "verified": 0, "unverified": [], "confidence": 100.0}

        verified = 0
        unverified = []
        for page, filename in citations:
            marker = f"[Page {page} of '{filename}']"
            if marker in context_text:
                verified += 1
            else:
                ref = f"Page {page} of '{filename}'"
                if ref not in unverified:
                    unverified.append(ref)

        return {
            "total": total,
            "verified": verified,
            "unverified": unverified,
            "confidence": round((verified / total) * 100.0, 1),
        }

    def _run_self_critique(self, context_text, query, answer):
        """
        Score the answer (0-10) for truthfulness and thoroughness.
        Uses the primary LLM or the judge chain depending on config critique_llm.
        Returns a dict with keys: score, reasoning, hallucination_rate.
        """
        critique_llm_cfg = CONFIG.get('validation', {}).get('critique_llm', 'primary')
        if critique_llm_cfg == 'judge':
            from llm_factory import get_judge_llm
            llm = get_judge_llm()
            label = "self-critique (judge)"
        else:
            llm = self.llm
            label = "self-critique (primary)"

        critique_template = self.jinja_env.get_template('auditor_critique.j2')
        validation_prompt = critique_template.render(
            context=context_text, query=query, answer=answer
        )
        try:
            critique_response = self._invoke_with_retry(
                llm, [HumanMessage(content=validation_prompt)], label=label
            )
            content = critique_response.content.strip()
            if content.startswith("```json"):
                content = content[7:].rstrip("` \n").strip()
            elif content.startswith("```"):
                content = content[3:].rstrip("` \n").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Self-critique failed: {e}")
            return {'score': 0, 'reasoning': f"Error: {e}", 'hallucination_rate': 0.0}

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
        if not hasattr(self, '_cross_llm_queue'):
            from llm_factory import get_judge_llm, get_fallback_judge_llm, get_secondary_llm, get_llm as _get_primary
            self._cross_llm_queue = [get_judge_llm, get_fallback_judge_llm, get_secondary_llm, _get_primary]
            self._cross_llm_idx   = 0
            self._cross_llm_llm   = None

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

        while self._cross_llm_idx < len(self._cross_llm_queue):
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
                response = self._invoke_with_retry(
                    self._cross_llm_llm, [HumanMessage(content=prompt)],
                    max_retries=2, base_delay=5, label="cross-LLM critique"
                )
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:].rstrip("` \n").strip()
                elif content.startswith("```"):
                    content = content[3:].rstrip("` \n").strip()
                result = json.loads(content)
                return {
                    "hallucinated":         bool(result.get("hallucinated", False)),
                    "unsupported_claims":   result.get("unsupported_claims", []),
                    "confidence_in_answer": int(result.get("confidence_in_answer", 50)),
                }
            except Exception as e:
                print(f"Cross-LLM provider failed ({str(e)[:100]}). Advancing to next provider...")
                self._cross_llm_llm = None
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

    def _decompose_query(self, design_assessment: str, test_procedure: str):
        """
        Ask the primary LLM to break a compound audit control into 2-3 focused,
        independent sub-questions suitable for separate document retrieval.

        Returns a list of sub-query strings, or None if decomposition fails so
        the caller can fall back to single-query retrieval.
        """
        prompt = (
            "You are an IFRS 9 audit assistant. Decompose the following audit control "
            "into 2-3 focused, independent sub-questions. Each sub-question must target "
            "a single, distinct retrievable fact (e.g. a specific methodology, threshold, "
            "governance document, or empirical result). If the control is already a single "
            "focused question, return just 1 sub-question.\n"
            "Return ONLY a valid JSON array of strings — no other text.\n\n"
            f"Design Assessment: {design_assessment}\n"
            f"Test Procedure: {test_procedure}"
        )
        try:
            response = self._invoke_with_retry(
                self.llm, [HumanMessage(content=prompt)], label="query decomposition"
            )
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:].rstrip("` \n").strip()
            elif content.startswith("```"):
                content = content[3:].rstrip("` \n").strip()
            sub_queries = json.loads(content)
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                return [q for q in sub_queries[:3] if q.strip()]
        except Exception as e:
            print(f"  [Decomposition] Failed ({e}). Using single query.")
        return None

    def _retrieve_decomposed(self, sub_queries: list, fallback_query: str, k: int = 10) -> list:
        """
        Retrieve for each sub-query independently, then merge and deduplicate by
        content fingerprint. Falls back to single-query retrieval if nothing is found.
        """
        seen = set()
        merged = []
        for sq in sub_queries:
            for doc in self.rag_engine.retrieve(sq, k=k):
                key = doc.page_content[:200]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged if merged else self.rag_engine.retrieve(fallback_query, k=k)

    # -------------------------------------------------------------------------
    # Main audit method
    # -------------------------------------------------------------------------

    def process_row(self, row):
        val_cfg = CONFIG.get('validation', {})
        rag_cfg = CONFIG.get('rag_settings', {})

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

        # --- Step 1: Initial retrieval and answer generation ---
        if rag_cfg.get('use_query_decomposition', False):
            sub_queries = self._decompose_query(design_assessment, test_procedure)
            if sub_queries:
                print(f"  [Decomposition] {len(sub_queries)} sub-queries for {control_ref}")
                retrieved_docs = self._retrieve_decomposed(sub_queries, query, k=10)
            else:
                retrieved_docs = self.rag_engine.retrieve(query, k=10)
        else:
            retrieved_docs = self.rag_engine.retrieve(query, k=10)

        num_retrieved = len(retrieved_docs)
        context_text, evidence_dict = self._build_context(retrieved_docs)

        response_template = self.jinja_env.get_template('auditor_response.j2')
        prompt_text = response_template.render(
            context=context_text,
            design_assessment=design_assessment,
            test_procedure=test_procedure,
        )
        response = self._invoke_with_retry(
            self.llm, [HumanMessage(content=prompt_text)], label="answer generation"
        )
        final_answer, extracted_evidence, compliance_verdict, classification = \
            self._parse_response(response.content)

        retrieval_expanded = False
        revision_count = 0

        # --- Step 2: Expanded retrieval on "Insufficient Info" ---
        # If the model couldn't find evidence, widen the retrieval net before
        # running any critiques or revisions.
        if compliance_verdict == "Insufficient Info":
            expanded_threshold = rag_cfg.get('retrieval_score_threshold', 1.8) + 0.4
            expanded_top_k = rag_cfg.get('client_top_k', 6) + 4
            print(
                f"  [Remediation] Insufficient Info — expanding retrieval for {control_ref} "
                f"(threshold {expanded_threshold:.1f}, top_k {expanded_top_k})..."
            )
            expanded_docs = self.rag_engine.retrieve(
                query, k=10,
                threshold_override=expanded_threshold,
                client_top_k_override=expanded_top_k,
            )
            if expanded_docs:
                context_text, evidence_dict = self._build_context(expanded_docs)
                num_retrieved = len(expanded_docs)
                response = self._invoke_with_retry(
                    self.llm,
                    [HumanMessage(content=response_template.render(
                        context=context_text,
                        design_assessment=design_assessment,
                        test_procedure=test_procedure,
                    ))],
                    label="answer generation (expanded retrieval)",
                )
                final_answer, extracted_evidence, compliance_verdict, classification = \
                    self._parse_response(response.content)
                retrieval_expanded = True

        # --- Step 3: Self-critique ---
        validation_result = {'score': 0, 'reasoning': 'Not run', 'hallucination_rate': 0.0}
        if val_cfg.get('enable_self_critique', True):
            validation_result = self._run_self_critique(context_text, query, final_answer)

        critique_score = validation_result.get('score', 0)

        # --- Step 4: Critique-Revise loop ---
        # If the judge scores the answer below the threshold, feed the critique
        # back to the primary LLM and ask it to revise. Keep the best answer
        # found across all attempts.
        retry_threshold = int(val_cfg.get('self_critique_retry_threshold', 6))
        max_revisions = int(val_cfg.get('max_revision_attempts', 2))

        if val_cfg.get('enable_self_critique', True) and critique_score < retry_threshold:
            revision_template = self.jinja_env.get_template('auditor_revision.j2')

            best = (final_answer, extracted_evidence, compliance_verdict,
                    critique_score, validation_result, classification)

            while critique_score < retry_threshold and revision_count < max_revisions:
                revision_count += 1
                print(
                    f"  [Remediation] Score {critique_score}/10 < {retry_threshold} — "
                    f"revising answer for {control_ref} "
                    f"(attempt {revision_count}/{max_revisions})..."
                )
                try:
                    revision_prompt = revision_template.render(
                        context=context_text,
                        design_assessment=design_assessment,
                        test_procedure=test_procedure,
                        previous_answer=final_answer,
                        critique_reasoning=validation_result.get('reasoning', ''),
                    )
                    rev_response = self._invoke_with_retry(
                        self.llm, [HumanMessage(content=revision_prompt)],
                        label=f"revision {revision_count}",
                    )
                    final_answer, extracted_evidence, compliance_verdict, _ = \
                        self._parse_response(rev_response.content)
                    validation_result = self._run_self_critique(context_text, query, final_answer)
                    critique_score = validation_result.get('score', 0)

                    if critique_score > best[3]:
                        best = (final_answer, extracted_evidence, compliance_verdict,
                                critique_score, validation_result, classification)

                except Exception as e:
                    print(f"  [Remediation] Revision {revision_count} failed: {e}")
                    break

            # Accept the best answer found across all attempts
            final_answer, extracted_evidence, compliance_verdict, critique_score, \
                validation_result, classification = best
            if revision_count > 0:
                print(
                    f"  [Remediation] Best score after {revision_count} revision(s): "
                    f"{critique_score}/10 for {control_ref}"
                )

        # --- Step 5: Citation verification ---
        # Check that every [Page X of 'Filename'] in the answer actually appeared
        # in the retrieved context (catches hallucinated page numbers).
        citation_check = self._verify_citations(final_answer, context_text)
        if citation_check['unverified']:
            print(
                f"  [Citation] {len(citation_check['unverified'])} unverified citation(s) "
                f"for {control_ref}: {citation_check['unverified']}"
            )

        # --- Step 6: Cross-LLM hallucination critique ---
        cross_result = None
        if val_cfg.get('enable_cross_llm_critique', True):
            cross_result = self._cross_llm_critique(context_text, final_answer)

        # --- Step 7: Confidence decomposition ---
        # Three independent sub-scores measuring different quality dimensions:
        #   Retrieval_Confidence  — enough context found?  (retrieved doc count vs target)
        #   Answer_Confidence     — model reasoned well?   (self-critique score × 10)
        #   Citation_Confidence   — citations grounded?    (verified / total citations)
        rerank_top_k_cfg = rag_cfg.get('rerank_top_k', 7)
        retrieval_confidence = round(
            min(num_retrieved / max(rerank_top_k_cfg, 1), 1.0) * 100.0, 1
        )
        answer_confidence = round((critique_score / 10.0) * 100.0, 1)
        citation_confidence_val = citation_check['confidence']

        if val_cfg.get('enable_self_critique', True):
            base_confidence = (
                0.5 * answer_confidence +
                0.3 * citation_confidence_val +
                0.2 * retrieval_confidence
            )
        else:
            # Without self-critique the answer sub-score is absent; redistribute weights
            base_confidence = (
                0.6 * citation_confidence_val +
                0.4 * retrieval_confidence
            )

        if cross_result is not None:
            cross_conf = float(cross_result.get('confidence_in_answer', 50))
            if cross_result.get('hallucinated', False):
                cross_conf = max(0.0, cross_conf - 30.0)
            confidence_score = round((base_confidence + cross_conf) / 2.0, 1)
        else:
            confidence_score = round(base_confidence, 1)

        # --- Step 8: Build result ---
        result = row.copy()
        result['Classification']        = classification
        result['AI_Answer']             = final_answer
        result['Evidence_Sources']      = extracted_evidence
        result['Verification_Step']     = extracted_evidence
        result['Compliance_Verdict']    = compliance_verdict
        result['Validation_Score']      = critique_score
        result['Validation_Reasoning']  = validation_result.get('reasoning', '')
        result['Hallucination_Rate']    = validation_result.get('hallucination_rate', 0.0)
        result['Confidence_Score']      = confidence_score
        result['Retrieval_Confidence_Score'] = retrieval_confidence
        result['Answer_Confidence_Score']    = answer_confidence
        result['Citation_Confidence']        = citation_confidence_val
        result['Unverified_Citations']       = "; ".join(citation_check['unverified'])
        result['Cross_LLM_Hallucinated'] = (
            cross_result.get('hallucinated', False) if cross_result else None
        )
        result['Cross_LLM_Concerns']   = (
            "; ".join(cross_result.get('unsupported_claims', [])) if cross_result else ""
        )
        result['Revision_Count']        = revision_count
        result['Retrieval_Expanded']    = retrieval_expanded

        return result
