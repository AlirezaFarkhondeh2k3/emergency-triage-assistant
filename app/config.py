from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "app" / "models"

# directories with your saved models
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
TRANSFORMER_ARTIFACTS_DIR = MODELS_DIR / "transformer_artifacts"
RAG_ARTIFACTS_DIR = MODELS_DIR / "rag_artifacts"

# toggle transformer vs baseline
USE_TRANSFORMER = False

CRISIS_CATEGORIES = [
    "flood",
    "fire",
    "earthquake",
    "storm",
    "landslide",
    "other",
]

SEVERITY_LEVELS = ["low", "medium", "high"]
