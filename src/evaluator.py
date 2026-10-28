"""
RAG Evaluation Subsystem — The RAG Triad.
Measures Context Relevance (retrieval quality), Faithfulness (groundedness & hallucination risk),
and Answer Relevance (query satisfaction).
"""

import re
import json
from typing import List, Dict, Any, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    from src.document_loader import Document


def split_into_statements(text: str) -> List[str]:
    """Break an answer into individual sentence-level claims."""
    clean = re.sub(r"\[Doc:[^\]]+\]", "", text)
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    valid_sentences = [
        s.strip() for s in sentences
        if len(s.strip()) > 15 and not s.strip().startswith("Based on the")
    ]
    return valid_sentences


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance using automated reference-free metrics.
    Supports both fast heuristic semantic verification and LLM-as-a-Judge.
    """

    def __init__(self, llm_judge: Optional[Any] = None):
        self.llm_judge = llm_judge

    def evaluate_context_relevance(
        self, query: str, retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Evaluate the signal-to-noise ratio in retrieved context chunks.
        """
        if not retrieved_docs:
            return {"score": 0.0, "reason": "No documents retrieved."}

        query_terms = set(
            w.lower() for w in re.findall(r"\w+", query)
            if len(w) > 3
        )
        if not query_terms:
            return {"score": 1.0, "reason": "Short query; cannot compute lexical relevance."}

        relevant_count = 0
        per_chunk_scores = []

        for doc in retrieved_docs:
            content_lower = doc.page_content.lower()
            matched = sum(1 for term in query_terms if term in content_lower)
            ratio = matched / len(query_terms)
            per_chunk_scores.append(round(ratio, 2))
            if ratio >= 0.3:
                relevant_count += 1

        score = round(relevant_count / len(retrieved_docs), 2)

        return {
            "score": score,
            "relevant_chunks": relevant_count,
            "total_chunks": len(retrieved_docs),
            "per_chunk_relevance": per_chunk_scores,
            "reason": f"{relevant_count} out of {len(retrieved_docs)} chunks contain key query concepts."
        }

    def evaluate_faithfulness(
        self, answer: str, context_text: str
    ) -> Dict[str, Any]:
        """
        Evaluate whether claims made in the answer are strictly supported by the context.
        Flags hallucinations and calculates Hallucination Risk Score.
        """
        if "cannot find the answer" in answer.lower():
            return {
                "score": 1.0,
                "hallucination_risk": 0.0,
                "hallucination_detected": False,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_statements": [],
                "reason": "Model correctly abstained from answering due to missing context."
            }

        # If LLM Judge is present, run LLM-as-a-Judge prompt
        if self.llm_judge is not None:
            try:
                judge_prompt = f"""You are an automated evaluation judge for a RAG system.
Verify whether each claim in the answer is supported by the context.

Context:
{context_text[:2000]}

Generated Answer:
{answer}

Respond ONLY with a JSON object in this format:
{{
  "total_claims": <number>,
  "supported_claims": <number>,
  "faithfulness_score": <float between 0.0 and 1.0>,
  "hallucination_detected": <true or false>,
  "unsupported_statements": [<list of strings>],
  "reasoning": "<short sentence>"
}}
"""
                response = self.llm_judge.invoke(judge_prompt)
                content = response.content if hasattr(response, "content") else str(response)
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    score = float(data.get("faithfulness_score", 1.0))
                    data["hallucination_risk"] = round(1.0 - score, 2)
                    return data
            except Exception as e:
                print(f"[Warning] LLM judge evaluation failed ({e}). Using heuristic claim verification.")

        # Heuristic claim verification
        statements = split_into_statements(answer)
        if not statements:
            return {
                "score": 1.0,
                "hallucination_risk": 0.0,
                "hallucination_detected": False,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_statements": [],
                "reason": "No distinct claims to verify."
            }

        context_lower = context_text.lower()
        supported = 0
        unsupported = []

        for stmt in statements:
            stmt_words = [w.lower() for w in re.findall(r"\w+", stmt) if len(w) > 3]
            if not stmt_words:
                supported += 1
                continue

            matches = sum(1 for w in stmt_words if w in context_lower)
            coverage = matches / len(stmt_words)

            if coverage >= 0.5:
                supported += 1
            else:
                unsupported.append(stmt)

        faithfulness = round(supported / len(statements), 2)
        hallucination_detected = len(unsupported) > 0
        hallucination_risk = round(1.0 - faithfulness, 2)

        return {
            "score": faithfulness,
            "hallucination_risk": hallucination_risk,
            "hallucination_detected": hallucination_detected,
            "total_claims": len(statements),
            "supported_claims": supported,
            "unsupported_statements": unsupported,
            "reason": (
                "All statements verified in context."
                if not unsupported
                else f"{len(unsupported)} claim(s) lacked conclusive grounding in context."
            )
        }

    def evaluate_answer_relevance(
        self, query: str, answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate whether the generated response directly answers the user's prompt.
        """
        if not answer.strip():
            return {"score": 0.0, "reason": "Empty answer generated."}

        if "cannot find the answer" in answer.lower():
            return {
                "score": 0.9,
                "reason": "Model appropriately indicated lack of sufficient context."
            }

        query_terms = set(
            w.lower() for w in re.findall(r"\w+", query)
            if len(w) > 3
        )
        if not query_terms:
            return {"score": 1.0, "reason": "Query too short for keyword analysis."}

        answer_lower = answer.lower()
        matched = sum(1 for term in query_terms if term in answer_lower)
        overlap_ratio = matched / len(query_terms)

        len_penalty = 1.0
        if len(answer) < 30:
            len_penalty = 0.6
        elif len(answer) > 3000:
            len_penalty = 0.85

        score = round(min(1.0, (overlap_ratio * 0.7 + 0.3) * len_penalty), 2)

        return {
            "score": score,
            "query_terms_matched": matched,
            "total_query_terms": len(query_terms),
            "reason": f"Answer directly addresses {matched} of {len(query_terms)} query concepts."
        }

    def run_full_evaluation(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document],
        context_text: str
    ) -> Dict[str, Any]:
        """
        Executes the full RAG Triad evaluation.
        """
        c_rel = self.evaluate_context_relevance(query, retrieved_docs)
        faith = self.evaluate_faithfulness(answer, context_text)
        a_rel = self.evaluate_answer_relevance(query, answer)

        composite_score = round(
            (float(c_rel["score"]) * 0.3) + (float(faith["score"]) * 0.4) + (float(a_rel["score"]) * 0.3),
            2
        )

        return {
            "composite_score": float(composite_score),
            "context_relevance": c_rel,
            "faithfulness": faith,
            "answer_relevance": a_rel,
            "hallucination_risk": float(faith.get("hallucination_risk", 0.0)),
            "hallucination_detected": bool(faith.get("hallucination_detected", False))
        }
