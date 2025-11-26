import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "event_aware_en"
RAW_DIR = Path(os.getenv("DATASET_ROOT", DEFAULT_RAW_DIR))

TSV_FILES = [
    "crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_train.tsv",
    "crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_dev.tsv",
    "crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_test.tsv",
]

# Lightweight fallback rows so CI can run even when the full dataset is absent.
SAMPLE_ROWS = [
    {
        "text": "Severe floods have blocked roads and displaced families.",
        "event": "flood",
        "lang": "en",
        "class_label": "informative",
    },
    {
        "text": "Heavy rain causing flood warnings across the valley.",
        "event": "flood",
        "lang": "en",
        "class_label": "informative",
    },
    {
        "text": "Wildfire approaching suburbs, evacuations underway.",
        "event": "wildfire",
        "lang": "en",
        "class_label": "informative",
    },
    {
        "text": "Fire crews contain brushfire near the highway.",
        "event": "wildfire",
        "lang": "en",
        "class_label": "non_informative",
    },
    {
        "text": "Aftershocks continue after yesterday's earthquake downtown.",
        "event": "earthquake",
        "lang": "en",
        "class_label": "informative",
    },
    {
        "text": "Earthquake tremor felt but no major damage reported.",
        "event": "earthquake",
        "lang": "en",
        "class_label": "non_informative",
    },
    {
        "text": "No damages reported after the small tornado passed.",
        "event": "tornado",
        "lang": "en",
        "class_label": "non_informative",
    },
    {
        "text": "Tornado warnings issued as storm system strengthens.",
        "event": "tornado",
        "lang": "en",
        "class_label": "informative",
    },
]


def map_event_to_category(event: str) -> str:
    """Map raw event string to a clean crisis category."""
    if not isinstance(event, str):
        return "other"

    e = event.lower()

    if "earthquake" in e:
        return "earthquake"
    if "flood" in e:
        return "flood"
    if "hurricane" in e or "typhoon" in e or "cyclone" in e:
        return "hurricane_typhoon_cyclone"
    if "tornado" in e:
        return "tornado"
    if "explosion" in e:
        return "explosion"
    if "wildfire" in e or "bushfire" in e or "fire" in e:
        return "fire"
    if "landslide" in e:
        return "landslide"
    if "ebola" in e or "syndrome" in e:
        return "epidemic"
    if "crash" in e or "collapse" in e:
        return "crash_collapse"

    return "other"


def load_raw_data() -> pd.DataFrame:
    """
    Load TSV files from RAW_DIR if present; otherwise, fall back to small inline data
    so CI can train and test without the large dataset.
    """
    if RAW_DIR.exists():
        dfs = []
        for name in TSV_FILES:
            path = RAW_DIR / name
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected TSV file missing: {path}. "
                    "Set DATASET_ROOT to the directory containing the CrisisBench TSV files."
                )
            print(f"Loading {path}")
            df = pd.read_csv(path, sep="\t")
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    print(
        f"RAW_DIR {RAW_DIR} not found. "
        "Using small built-in sample to keep CI green. "
        "Set DATASET_ROOT to point to the full dataset for real training."
    )
    return pd.DataFrame(SAMPLE_ROWS)


def main():
    data = load_raw_data()
    print("Raw shape:", data.shape)

    # Keep only English text and drop missing text
    data = data[data["lang"] == "en"].copy()
    data = data.dropna(subset=["text", "event"])
    print("After lang/text filter:", data.shape)

    # Create the clean crisis category
    data["category"] = data["event"].apply(map_event_to_category)

    # Optional: drop rows mapped to "other" if you want a stricter task
    # data = data[data["category"] != "other"]

    # Keep only the columns we care about
    data = data[["text", "event", "category", "class_label"]]

    out_path = ROOT / "data" / "processed" / "incidents.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    print("Saved cleaned dataset to:", out_path)
    print("Category counts:")
    print(data["category"].value_counts())


if __name__ == "__main__":
    main()
