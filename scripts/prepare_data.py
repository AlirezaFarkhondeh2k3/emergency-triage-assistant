from pathlib import Path
import os
import pandas as pd   # keep whatever you already had

# Repo root (…/emergency-triage-assistant)
ROOT = Path(__file__).resolve().parents[1]

# If DATASET_ROOT is set, use that.
# Otherwise default to repo-relative path: data/raw/event_aware_en
RAW_DIR = Path(
    os.getenv(
        "DATASET_ROOT",
        ROOT / "data" / "raw" / "event_aware_en"
    )
)

TSV_FILES = [
    RAW_DIR / "crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_train.tsv",
    RAW_DIR / "crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_dev.tsv",
    RAW_DIR / "crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_test.tsv",
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


def main():
    # 2. Load and concatenate the three splits
    dfs = []
    for name in TSV_FILES:
        path = RAW_DIR / name
        print(f"Loading {path}")
        df = pd.read_csv(path, sep="\t")
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print("Raw shape:", data.shape)

    # 3. Keep only English text and drop missing text
    data = data[data["lang"] == "en"].copy()
    data = data.dropna(subset=["text", "event"])
    print("After lang/text filter:", data.shape)

    # 4. Create the clean crisis category
    data["category"] = data["event"].apply(map_event_to_category)

    # Optional: drop rows mapped to "other" if you want a stricter task
    # data = data[data["category"] != "other"]

    # 5. Keep only the columns we care about
    data = data[["text", "event", "category", "class_label"]]

    # 6. Save to repo-relative processed path
    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "incidents.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    print("Saved cleaned dataset to:", out_path)
    print("Category counts:")
    print(data["category"].value_counts())


if __name__ == "__main__":
    main()
