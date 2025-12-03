from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import requests

from app.models.severity_llm import SeverityLabel

logger = logging.getLogger(__name__)


@dataclass
class LLMReplyContext:
    category: str
    severity: SeverityLabel
    location: Optional[str]
    guidance: str
    summary: str
    location_known: bool = False
    people_known: bool = False
    safety_known: bool = False
    triage_complete: bool = False


@dataclass
class ReplyResult:
    text: str
    triage_complete: bool


class LLMReplyModel:
    def __init__(self, base_url: str | None = None, model_name: str | None = None):
        self.base_url = base_url or "http://localhost:11434"
        self.model_name = model_name or "llama3"
        self.triage_complete = False

    def _build_prompt(self, messages: List[Dict[str, str]], ctx: LLMReplyContext) -> Tuple[str, bool]:
        recent = messages[-6:] if len(messages) > 6 else messages
        convo_lines = []
        last_user = ""
        for m in recent:
            role = m.get("role", "")
            content = m.get("content", "")
            if not content:
                continue
            prefix = "Assistant: " if role == "assistant" else "User: "
            convo_lines.append(prefix + content)
            if role == "user":
                last_user = content
        convo_text = "\n".join(convo_lines)

        safety_confirmed_text = (ctx.summary or "") + " " + (last_user or "")
        safety_confirmed = any(
            phrase in safety_confirmed_text.lower()
            for phrase in [
                "i am safe",
                "i'm safe",
                "we are safe",
                "we're safe",
                "we are both safe",
                "safe upstairs",
                "safe now",
                "away from danger",
                "out of danger",
                "away from the danger",
            ]
        )
        effective_safety = ctx.safety_known or safety_confirmed

        triage_complete = ctx.location_known and ctx.people_known and effective_safety

        prompt = (
            "You are an emergency triage assistant. You always base your answer on the category, severity, location, "
            "guidance, summary from ctx and the full user message history. You must speak in short, focused paragraphs, "
            "no markdown lists.\n\n"
            "Information-gathering phase:\n"
            "- Ask at most the following follow-up questions, and each at most once:\n"
            "  Missing location: ask for address or nearby landmark.\n"
            "  Missing people affected: ask how many people are affected or injured.\n"
            "  Safety unknown: ask if they are currently in a safe place away from the immediate danger.\n"
            "- Only ask a follow-up if that piece of info is clearly missing or ambiguous in the conversation so far.\n"
            "Completion rule - very important:\n"
            "- If you have a location (address or clear landmark), you know roughly how many people are affected, and "
            "the user has said they are safe or away from danger, then do not ask any more questions and do not repeat "
            "earlier questions. Give concise guidance plus a final confirmation like: \"Thank you, your report has been "
            "submitted. A trained responder is reviewing the information and help is on the way. Please stay safe and "
            "follow any instructions from local authorities.\" Never end your message with another question when those "
            "three are known.\n"
            "- Even if information is incomplete, never repeat the same follow-up question more than once. If the user "
            "ignores a question, continue with guidance and gently suggest contacting emergency services directly.\n\n"
            "Context:\n"
            f"- Category: {ctx.category}\n"
            f"- Severity: {ctx.severity}\n"
            f"- Location: {ctx.location or 'not provided'}\n"
            f"- Summary: {ctx.summary}\n"
            f"- Guidance hint: {ctx.guidance}\n"
            f"- Location known: {ctx.location_known}\n"
            f"- People known: {ctx.people_known}\n"
            f"- Safety known: {effective_safety}\n\n"
            "Recent conversation:\n"
            f"{convo_text}\n\n"
        )

        followup_question = None
        if not triage_complete:
            if not ctx.location_known:
                followup_question = "Where exactly are you right now? Please give an address or nearby landmark."
            elif not ctx.people_known:
                followup_question = "How many people, including you, are affected or injured?"
            elif not effective_safety:
                followup_question = "Are you currently in a safe place away from the immediate danger, or still close to the fire/flood/incident?"

        if triage_complete:
            prompt += (
                "All required details are present. Provide a concise guidance paragraph using the guidance hint and end with this "
                "confirmation (plain sentences, no markdown lists): Thank you, your report has been submitted. A trained responder is "
                "reviewing the information and help is on the way. Please stay safe and follow any instructions from local authorities.\n"
            )
        else:
            prompt += (
                "Write one concise reply in short paragraphs (no markdown lists). Acknowledge the user briefly, provide tailored safety "
                "guidance using the guidance hint, and only ask one follow-up question if a critical detail is missing or ambiguous:\n"
                f"{followup_question}\n"
                "Do not repeat a follow-up question that has already been asked in the conversation. If you choose not to ask a question "
                "because it was already asked or ignored, still provide guidance and reassure them that a responder is reviewing the case, "
                "and suggest contacting emergency services directly if needed.\n"
            )

        prompt += "\nAssistant reply:"
        return prompt, triage_complete

    def generate_reply(self, messages: List[Dict[str, str]], ctx: LLMReplyContext) -> Optional[ReplyResult]:
        prompt, triage_complete = self._build_prompt(messages, ctx)
        if not prompt:
            return None

        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=20)
        except Exception as exc:
            logger.warning("Ollama request failed: %r", exc)
            return None

        if resp.status_code != 200:
            logger.warning("Ollama failed: status %s, body: %s", resp.status_code, resp.text[:200])
            return None

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Ollama JSON parse failed: %r", exc)
            return None

        reply = data.get("response", "") if isinstance(data, dict) else ""
        reply = reply.strip()
        if not reply or len(reply) < 20:
            logger.warning("Ollama reply too short or empty")
            return None

        if triage_complete:
            ctx.triage_complete = True
            self.triage_complete = True

        logger.info("Ollama reply length: %d", len(reply))
        return ReplyResult(text=reply, triage_complete=triage_complete)
