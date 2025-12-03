from __future__ import annotations

import logging
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class ChatSummarizer:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model_name = "llama3"

    def summarize(self, user_messages: List[str]) -> Optional[str]:
        if not user_messages:
            return None

        transcript = "\n".join(m for m in user_messages if m)
        prompt = (
            "You are an emergency triage assistant.\n"
            "Read the user messages below and write a single 1-2 sentence summary of the situation, "
            "focusing on what is happening, where, and how serious it sounds. "
            'Do not mention "severity" labels. Do not talk about being an AI.\n\n'
            f"User messages:\n{transcript}\n\n"
            "Summary:\n"
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
            logger.warning("Ollama summarizer request failed: %r", exc)
            return None

        try:
            data = resp.json()
            summary = data.get("response", "") if isinstance(data, dict) else ""
        except Exception as exc:
            logger.warning("Ollama summarizer JSON parse failed: %r", exc)
            return None

        summary = summary.strip()
        if len(summary) < 10:
            logger.warning("Ollama summarizer returned too-short summary")
            return None

        logger.info("Ollama summarizer used (len=%d)", len(summary))
        return summary
