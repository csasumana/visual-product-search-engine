import pandas as pd

from app.core.config import settings


SUBSET_SIZE = 25000
OUTPUT_PATH = "data/metadata_subset_25k.csv"


def main():
    metadata_path = settings.resolve_path(settings.METADATA_PATH)

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    print(f"[INFO] Loading full metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)

    if len(df) < SUBSET_SIZE:
        print(f"[WARNING] Dataset has only {len(df)} rows, less than {SUBSET_SIZE}. Using full dataset.")
        subset_df = df.copy()
    else:
        # random sample for broad coverage
        subset_df = df.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)

    output_path = settings.resolve_path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(output_path, index=False)

    print("\n=== SUBSET CREATED ===")
    print(f"Subset rows: {len(subset_df)}")
    print(f"Saved to: {output_path}")
    print(f"Unique categories in subset: {subset_df['category'].nunique()}")


if __name__ == "__main__":
    main()