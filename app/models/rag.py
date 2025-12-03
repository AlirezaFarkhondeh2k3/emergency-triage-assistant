import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import RAG_ARTIFACTS_DIR


class RAGGuidance:
    """
    Very light weight knowledge base lookup:
      - loads kb.json from RAG_ARTIFACTS_DIR
      - matches on category and severity
      - returns a guidance text snippet
    """

    def __init__(self):
        kb_path: Path = RAG_ARTIFACTS_DIR / "kb.json"
        if kb_path.exists():
            with kb_path.open("r", encoding="utf-8") as f:
                self.kb: List[Dict[str, Any]] = json.load(f)
        else:
            self.kb = []

    def _match_doc(self, category: str, severity: str) -> str:
        # expect docs like {"category": "flood", "severity": "high", "text": "..."}
        for doc in self.kb:
            if doc.get("category") == category and doc.get("severity") == severity:
                return doc.get("text", "")
        # fallback: category only
        for doc in self.kb:
            if doc.get("category") == category:
                return doc.get("text", "")
        return ""

    def generate_guidance(self, summary: str, category: str, severity: str) -> str:
        kb_text = self._match_doc(category, severity)
        if not kb_text:
            kb_text = (
                "Stay safe, avoid unnecessary risk, follow instructions from local "
                "authorities, and contact emergency services in urgent situations."
            )

        return (
            f"Based on your description, this appears to be a {severity} severity "
            f"{category} situation. {kb_text}"
        )


class IncidentRAG:
    """
    Wrapper used by TriagePipeline.

    This gives the pipeline a stable interface (IncidentRAG) while letting
    the implementation be a simple knowledge base lookup for now. Later you
    can swap RAGGuidance for a real RAG stack without touching the pipeline.
    """

    def __init__(self):
        self._guidance = RAGGuidance()

    def generate_guidance(
        self,
        summary: str,
        category: str,
        severity: str,
    ) -> str:
        """
        Main entry point used by the pipeline.

        If tests or the API expect a guidance string, this method should be
        the one TriagePipeline calls.
        """
        return self._guidance.generate_guidance(summary, category, severity)

    # Optional ergonomic aliases in case you call it differently elsewhere.
    def __call__(
        self,
        summary: str,
        category: str,
        severity: str,
    ) -> str:
        return self.generate_guidance(summary, category, severity)
