from dataclasses import asdict, dataclass

from typing import Optional

from app.models.classifier import CrisisClassifier
from app.models.rag import IncidentRAG
from app.models.location_extractor import LocationExtractor
from app.models.severity_llm import LLMSeverityModel


@dataclass
class TriageResult:
    """Structured triage output used by the agent and API."""
    reply: str
    category: str
    severity: str
    location: str
    guidance: str
    summary: str

    def to_dict(self) -> dict:
        """Return a plain dict for FastAPI responses/tests."""
        return asdict(self)


class TriagePipeline:
    def __init__(self, use_transformer: bool = False, use_rag: bool = True) -> None:
        """
        High-level pipeline:
        - takes raw user text
        - uses CrisisClassifier to get category, severity, location
        - optionally uses RAG to get tailored guidance
        """
        # For now CrisisClassifier internally decides how to classify;
        # we accept use_transformer just to keep the API stable.
        self.use_transformer = use_transformer

        self.classifier = CrisisClassifier()
        self.rag = IncidentRAG() if use_rag else None
        self.location_extractor = LocationExtractor()
        self.severity_llm = LLMSeverityModel()

    def _summarize_text(self, text: str) -> str:
        """
        Lightweight summarization placeholder.

        Keeps behaviour deterministic for tests while giving downstream
        layers a short, human-readable summary.
        """
        summary = text.strip()
        return summary if summary else "User reported an issue."

    def run(self, text: str, summary: Optional[str] = None, last_user_text: Optional[str] = None) -> TriageResult:
        """
        Run full triage on the full conversation text:
        - classify (category, severity, location)
        - retrieve guidance from RAG (if available)
        - return a structured TriageResult
        """
        summary_text = summary or self._summarize_text(text)
        cls_result = self.classifier.classify(summary_text)
        severity = self._adjust_severity(cls_result.category, cls_result.severity, summary_text)
        llm_sev = self.severity_llm.infer(summary_text)
        if llm_sev:
            if llm_sev.severity == "high" and severity != "high":
                severity = "high"
            elif severity == "low" and llm_sev.severity == "medium":
                severity = "medium"

        guidance = (
            "Stay safe, avoid unnecessary risk, follow instructions from local "
            "authorities, and contact emergency services in urgent situations."
        )
        if self.rag is not None:
            guidance = self.rag.generate_guidance(
                summary=summary_text,
                category=cls_result.category,
                severity=severity,
            )

        location_hint = getattr(cls_result, "location", "") or ""
        if not location_hint and summary_text:
            location_hint = self._extract_location(summary_text)

        reply = (
            f"Understood. This looks like a {severity} severity "
            f"{cls_result.category} situation."
        )
        if location_hint:
            reply += f" Reported location: {location_hint}."
        reply += f" {guidance}"

        return TriageResult(
            reply=reply,
            category=cls_result.category,
            severity=severity,
            location=location_hint or "",
            guidance=guidance,
            summary=summary_text,
        )

    def _extract_location(self, text: str) -> str:
        """
        Minimal heuristic to pull a location hint from the conversation text.
        """
        extracted = self.location_extractor.extract(text)
        if extracted:
            return extracted.strip()
        return ""

    def _adjust_severity(self, category: str, base_severity: str, text: str) -> str:
        """
        Apply simple overrides to avoid underestimating obviously dangerous floods.
        """
        severity = base_severity or "medium"
        lowered = text.lower()

        # Universal high-risk cues (violence, severe medical, fire entrapment)
        high_any = [
            "gunshot",
            "gunshots",
            "shots fired",
            "active shooter",
            "shooter",
            "armed",
            "stabbing",
            "stabbed",
            "knife",
            "attack",
            "bleeding heavily",
            "heavy bleeding",
            "blood everywhere",
            "not breathing",
            "no pulse",
            "doing cpr",
            "trying cpr",
            "fire inside",
            "smoke is filling",
            "smoke filling",
            "cannot reach the exit",
            "can't reach the exit",
            "hallway is burning",
            "trapped inside",
            "can't get out",
            "cannot get out",
        ]

        if any(term in lowered for term in high_any):
            severity = "high"

        flood_terms = ["flood", "flooding", "water is rising", "water rising", "water rising quickly"]
        high_risk_terms = ["trapped", "swept", "swept away", "cannot breathe", "can't breathe", "unconscious"]

        if category == "flood" and any(term in lowered for term in flood_terms):
            if severity == "low":
                severity = "medium"
            if any(term in lowered for term in high_risk_terms):
                severity = "high"

        # Escalate to high for flood with trapped + child/people cues
        trapped_terms = [
            "stuck",
            "trapped",
            "cannot get out",
            "can't get out",
            "cannot exit",
            "can't exit",
            "we can’t leave",
            "we can't leave",
        ]
        child_terms = ["my son", "my daughter", "my kid", "my child", "children", "people are trapped"]
        if category == "flood" and any(t in lowered for t in trapped_terms) and any(c in lowered for c in child_terms):
            severity = "high"

        # Smoke-only high-risk escalation
        smoke_terms = [
            "heavy smoke",
            "thick smoke",
            "dense smoke",
            "hard to breathe",
            "cant breathe",
            "can't breathe",
            "struggling to breathe",
            "coughing badly",
            "choking on smoke",
        ]
        if any(term in lowered for term in smoke_terms):
            severity = "high"

        if severity not in ["low", "medium", "high"]:
            severity = "medium"

        return severity
