from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

import requests

SeverityLabel = Literal["low", "medium", "high"]

logger = logging.getLogger(__name__)


@dataclass
class LLMSeverityResult:
    severity: SeverityLabel
    reason: str


class LLMSeverityModel:
    def __init__(self, model_name: str | None = None, base_url: str | None = None):
        self.model_name = model_name or "llama3"
        self.base_url = base_url or "http://localhost:11434"

    def _rule_based_severity(self, text: str) -> LLMSeverityResult:
        """
        Deterministic fallback if LLM call fails.
        """
        t = text.lower()
        severity: SeverityLabel = "medium"
        reason = "Defaulted to medium."

        high_terms = [
            "unconscious",
            "not breathing",
            "no pulse",
            "cpr",
            "gunshot",
            "shooting",
            "explosion",
            "trapped",
            "collapsed",
            "heavy bleeding",
            "bleeding heavily",
            "cant get out",
            "can't get out",
        ]
        injured_terms = ["people injured", "injured", "people collapsed"]

        if any(term in t for term in high_terms):
            severity = "high"
            reason = "Rule-based escalation: life-threatening cue detected."
        elif ("fire" in t or "smoke" in t) and any(term in t for term in injured_terms):
            severity = "high"
            reason = "Rule-based escalation: fire/smoke with injuries."

        return LLMSeverityResult(severity=severity, reason=reason)

    def infer(self, text: str) -> LLMSeverityResult:
        prompt = (
            "You are an emergency triage assistant.\n"
            'You must classify the severity of this situation as "low", "medium", or "high".\n\n'
            "Situation:\n"
            f"{text}\n\n"
            "Definitions:\n"
            "- low: minor issue, no clear danger, no injuries or people trapped.\n"
            "- medium: serious problem that may become dangerous but no clear life-threatening signs yet.\n"
            "- high: likely or actual life-threatening emergency (injuries, unconscious, not breathing, heavy bleeding, trapped, big fire in an occupied building, gunshots, explosion, active violence).\n\n"
            'Answer in JSON ONLY:\n{"severity": "low|medium|high", "reason": "short explanation"}\n'
        )

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=1)
        except Exception as exc:
            logger.warning("Ollama severity request failed: %r", exc)
            return self._rule_based_severity(text)

        if resp.status_code != 200:
            logger.warning("Ollama severity failed: status %s, body: %s", resp.status_code, resp.text[:200])
            return self._rule_based_severity(text)

        try:
            data = resp.json()
            raw_response = data.get("response", "") if isinstance(data, dict) else ""
            parsed = json.loads(raw_response) if raw_response else {}
            sev = str(parsed.get("severity", "")).lower()
            reason = parsed.get("reason", "")
            if sev not in ("low", "medium", "high"):
                raise ValueError("Invalid severity from LLM")
            if not reason:
                reason = "LLM provided no reason."
            logger.info("Ollama severity reply used")
            return LLMSeverityResult(severity=sev, reason=reason)
        except Exception as exc:
            logger.warning("Ollama severity parsing failed: %r", exc)
            return self._rule_based_severity(text)
