from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RAGResult:
    category: str
    guidance: str
    retrieved_docs: List[Dict[str, Any]]


class IncidentRAG:
    """
    Embedding-based RAG over a small incident playbook.

    Artifacts are built by scripts/build_rag_index.py:
      - app/models/rag_artifacts/kb.json
      - app/models/rag_artifacts/embeddings.npy
    """

    def __init__(self) -> None:
        artifacts_dir = Path(__file__).resolve().parent / "rag_artifacts"
        kb_path = artifacts_dir / "kb.json"
        emb_path = artifacts_dir / "embeddings.npy"

        if not kb_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                f"RAG artifacts not found in {artifacts_dir}. "
                "Run scripts/build_rag_index.py first."
            )

        with kb_path.open("r", encoding="utf-8") as f:
            self.docs: List[Dict[str, Any]] = json.load(f)

        self.embeddings: np.ndarray = np.load(emb_path)

        # Same model used in build_rag_index
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def retrieve(
        self,
        text: str,
        category: str | None = None,
        top_k: int = 3,
    ) -> RAGResult:
        """
        Retrieve top_k guidance docs, optionally filtered by incident category.
        """
        # Encode query text
        query_emb = self.model.encode([text])
        sims = cosine_similarity(query_emb, self.embeddings)[0]

        # Sort by similarity descending
        idxs = np.argsort(-sims)

        selected: List[Dict[str, Any]] = []
        for i in idxs:
            doc = self.docs[i]
            if category is not None and doc.get("category") != category:
                continue
            selected.append(doc)
            if len(selected) >= top_k:
                break

        if not selected:
            guidance = (
                "No specific playbook entry found. "
                "Move away from immediate danger if possible and contact local emergency services."
            )
        else:
            # Simple aggregation: concatenate contents
            guidance = " ".join(d["content"] for d in selected)

        return RAGResult(
            category=category or "unknown",
            guidance=guidance,
            retrieved_docs=selected,
        )
