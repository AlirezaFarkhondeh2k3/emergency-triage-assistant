# app/agent.py

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import re

from app.pipeline import TriagePipeline
from app.config import USE_TRANSFORMER
from app.models.reply_llm import LLMReplyModel, LLMReplyContext
from app.models.summarizer import ChatSummarizer


@dataclass
class AgentResult:
    reply: str
    category: str
    severity: str
    location: Optional[str]
    guidance: str
    summary: str

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "category": self.category,
            "severity": self.severity,
            "location": self.location,
            "guidance": self.guidance,
            "summary": self.summary,
        }


class TriageAgent:
    """Chat-friendly wrapper over the triage pipeline."""
    def __init__(self, use_transformer: Optional[bool] = None):
        """
        Wrapper around TriagePipeline used by the API layer.

        `use_transformer` is passed from app.api (and tests) but we keep the
        default behaviour from config if it is not provided.
        """
        if use_transformer is None:
            use_transformer = USE_TRANSFORMER

        # Keep a flag in case you need it later (for debugging / logging)
        self.use_transformer = use_transformer

        # If your TriagePipeline.__init__ does NOT take `use_transformer`,
        # do NOT pass it. Let the pipeline read from config internally.
        self.pipeline = TriagePipeline()
        self.reply_llm = LLMReplyModel()
        self.summarizer = ChatSummarizer()

        # If, in your implementation, TriagePipeline.__init__ *does* accept
        # the flag, you can instead do:
        # self.pipeline = TriagePipeline(use_transformer=use_transformer)

    def _latest_user_message(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return next((m for m in reversed(messages) if m.get("role") == "user"), None)

    def _is_greeting(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        if not t:
            return False
        greeting_keywords = ("hi", "hello", "hey", "good morning", "good evening", "good afternoon")
        if any(t == g or t.startswith(g + " ") for g in greeting_keywords):
            emergency_keywords = (
                "fire",
                "smoke",
                "flood",
                "water rising",
                "earthquake",
                "gunshot",
                "bleeding",
                "not breathing",
                "no pulse",
                "trapped",
                "accident",
                "explosion",
                "storm",
                "landslide",
            )
            if not any(k in t for k in emergency_keywords):
                return True
        return False

    def _build_summary(self, messages: List[Dict[str, Any]]) -> str:
        """
        Build a short situation summary from user messages, preferring LLM summarizer.
        """
        user_texts = [m.get("content", "") for m in messages if m.get("role") == "user" and m.get("content")]
        llm_summary = None
        if self.summarizer is not None:
            try:
                llm_summary = self.summarizer.summarize(user_texts)
            except Exception:
                llm_summary = None

        if llm_summary:
            return llm_summary.strip()

        fallback = " ".join(user_texts).strip()
        if fallback:
            return fallback[-400:]
        return "User reported an issue."

    def _people_known(self, text: str) -> bool:
        if not text:
            return False
        if re.search(r"\b\d+\s+(people|persons|kids|children|adults|workers|passengers)\b", text, flags=re.I):
            return True
        if re.search(r"\b(alone|only me|by myself|just me)\b", text, flags=re.I):
            return True
        return False

    def _safety_known(self, text: str) -> Optional[bool]:
        if not text:
            return None
        low = text.lower()
        safe_cues = ["i'm safe", "im safe", "we are safe", "away from danger", "out of danger", "outside now", "safe now", "ok now"]
        unsafe_cues = ["not safe", "still inside", "still in danger", "still here", "can't get out", "cannot get out", "trapped", "stuck"]
        if any(cue in low for cue in safe_cues):
            return True
        if any(cue in low for cue in unsafe_cues):
            return False
        return None

    def _build_agent_result(self, messages: List[Dict[str, Any]]) -> AgentResult:
        if not messages:
            raise ValueError("No messages provided to TriageAgent.run")

        last_user_message = self._latest_user_message(messages)
        user_texts = [m.get("content", "") for m in messages if m.get("role") == "user" and m.get("content")]
        conversation_text = " ".join(user_texts).strip()
        last_user_text = last_user_message.get("content") if last_user_message else conversation_text

        summary = self._build_summary(messages)

        pipeline_result = self.pipeline.run(text=conversation_text, summary=summary, last_user_text=last_user_text)
        ctx = LLMReplyContext(
            category=pipeline_result.category,
            severity=pipeline_result.severity,
            location=pipeline_result.location or "",
            guidance=pipeline_result.guidance,
            summary=pipeline_result.summary,
            location_known=bool(pipeline_result.location),
            people_known=self._people_known(conversation_text),
            safety_known=bool(self._safety_known(conversation_text)) if self._safety_known(conversation_text) is not None else False,
            triage_complete=False,
        )

        greeting_reply = (
            "Hi, I’m an emergency triage assistant. Please describe what is happening and where you are, "
            "so I can help assess how urgent it is."
        )

        if self._is_greeting(last_user_text):
            reply = greeting_reply
        else:
            reply = ""
            # If triage already complete, keep responses short and avoid more follow-ups.
            if getattr(self.reply_llm, "triage_complete", False):
                reply = "Your report has already been submitted. Stay safe until responders arrive."
            elif self.reply_llm is not None:
                try:
                    llm_result = self.reply_llm.generate_reply(messages, ctx)
                    if llm_result and llm_result.text:
                        reply = llm_result.text
                        if llm_result.triage_complete:
                            self.reply_llm.triage_complete = True
                        print("LLM reply used")
                    else:
                        print("LLM reply failed: empty response")
                except Exception as e:
                    print(f"LLM reply failed: {e!r}")
                    reply = ""

            if not reply:
                reply = (
                    f"Understood. This looks like a {pipeline_result.severity} severity "
                    f"{pipeline_result.category} situation. "
                    f"Based on your description, this appears to be a {pipeline_result.severity} "
                    f"severity {pipeline_result.category} situation. "
                    "Stay safe, avoid unnecessary risk, follow instructions from local authorities, "
                    "and contact emergency services in urgent situations."
                )

        return AgentResult(
            reply=reply,
            category=pipeline_result.category,
            severity=pipeline_result.severity,
            location=pipeline_result.location or "",
            guidance=pipeline_result.guidance,
            summary=pipeline_result.summary,
        )

    def run(self, messages: List[Dict[str, Any]]) -> AgentResult:
        """
        Backwards-compatible entry point used by tests and legacy API code.
        """
        return self._build_agent_result(messages)

    def run_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Chat-style entry point:
        - accepts a list of {role, content} messages
        - uses the full conversation to classify/extract
        - returns a plain dict with reply + structured fields
        """
        result = self._build_agent_result(messages)
        return result.to_dict()
