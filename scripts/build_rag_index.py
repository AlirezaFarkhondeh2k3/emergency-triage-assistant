from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

KB_PATH = ROOT / "data" / "knowledge" / "incident_playbook.jsonl"
OUT_DIR = ROOT / "app" / "models" / "rag_artifacts"


def load_kb():
    """Load knowledge base from JSONL file."""
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {KB_PATH}")

    docs = []
    with KB_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    if not docs:
        raise ValueError("Knowledge base is empty. Add entries to incident_playbook.jsonl.")

    return docs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_kb()
    texts = [d["content"] for d in docs]

    print(f"Loaded {len(docs)} documents from KB")
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Encoding KB documents...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
    )

    embeddings_path = OUT_DIR / "embeddings.npy"
    kb_json_path = OUT_DIR / "kb.json"

    np.save(embeddings_path, embeddings)
    with kb_json_path.open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Saved KB metadata to: {kb_json_path}")


if __name__ == "__main__":
    main()
