from dataclasses import dataclass
from typing import Dict


@dataclass
class RAGResult:
    category: str
    guidance: str


class IncidentRAG:
    """
    Very simple RAG-style playbook.

    In a real system this would query a vector store or LLM.
    For now we return canned guidance text per incident category.
    """

    def __init__(self) -> None:
        # You can tweak this text later; keep keys in sync with classifier labels.
        self.playbook: Dict[str, str] = {
            "earthquake": (
                "If you are indoors, drop, cover, and hold on. "
                "Stay away from windows and heavy objects. After shaking stops, "
                "evacuate to open space and check for injuries and gas leaks."
            ),
            "flood": (
                "Move to higher ground immediately. Avoid walking or driving "
                "through flood waters. Do not enter flooded buildings until "
                "authorities declare them safe."
            ),
            "hurricane_typhoon_cyclone": (
                "Stay inside, away from windows. Follow local evacuation orders. "
                "Avoid coastal areas and be prepared for flooding and power loss."
            ),
            "tornado": (
                "Go to an interior room on the lowest floor, away from windows. "
                "If outside, lie flat in a low spot and cover your head."
            ),
            "landslide": (
                "Move away from the path of the slide to higher ground. "
                "Stay clear of valleys and river channels."
            ),
            "fire": (
                "Evacuate immediately using stairs, not elevators. "
                "Stay low under smoke and close doors behind you."
            ),
            "crash_collapse": (
                "Call emergency services. Keep a safe distance from unstable "
                "structures and watch for falling debris."
            ),
            "explosion": (
                "Get to a safe location, away from glass and damaged buildings. "
                "Do not touch suspicious objects. Follow instructions from authorities."
            ),
            "epidemic": (
                "Follow public health guidance. Maintain hygiene, wear masks if advised, "
                "and avoid crowded areas."
            ),
            # Fallback guidance
            "other": (
                "Describe the incident clearly to emergency services. "
                "Follow local authority instructions and prioritize your safety."
            ),
        }

    def retrieve(self, category: str) -> RAGResult:
        # Use specific guidance when available, otherwise fall back to 'other'.
        guidance = self.playbook.get(category, self.playbook["other"])
        return RAGResult(category=category, guidance=guidance)
