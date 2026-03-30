from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image

from app.core.config import settings


REQUIRED_COLUMNS = {"image_id", "image_path", "category"}


def validate_metadata_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")


def is_image_readable(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    metadata_path = settings.resolve_path(settings.METADATA_PATH)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_path}")

    print(f"[INFO] Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    validate_metadata_columns(df)

    print(f"[INFO] Total rows in metadata: {len(df)}")

    valid_rows = []
    failed_rows: List[dict] = []

    for idx, row in df.iterrows():
        image_id = row["image_id"]
        rel_image_path = str(row["image_path"])
        category = str(row["category"])
        title = row["title"] if "title" in df.columns else None
        coarse_label = row["coarse_label"] if "coarse_label" in df.columns else None

        full_path = settings.resolve_raw_dataset_path(rel_image_path)

        if not full_path.exists():
            failed_rows.append({
                "image_id": image_id,
                "image_path": rel_image_path,
                "category": category,
                "reason": "file_not_found",
            })
            continue

        if not is_image_readable(full_path):
            failed_rows.append({
                "image_id": image_id,
                "image_path": rel_image_path,
                "category": category,
                "reason": "unreadable_or_corrupt",
            })
            continue

        valid_rows.append({
            "image_id": image_id,
            "image_path": rel_image_path,
            "category": category,
            "title": title if pd.notna(title) else None,
            "coarse_label": int(coarse_label) if pd.notna(coarse_label) else None,
        })

        if (idx + 1) % 5000 == 0:
            print(f"[INFO] Validated {idx + 1}/{len(df)} rows...")

    valid_df = pd.DataFrame(valid_rows)
    failed_df = pd.DataFrame(failed_rows)

    valid_df.to_csv(metadata_path, index=False)

    failed_path = settings.resolve_path("data/failed_images.csv")
    failed_df.to_csv(failed_path, index=False)

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Valid rows:   {len(valid_df)}")
    print(f"Failed rows:  {len(failed_df)}")
    print(f"Cleaned metadata saved to: {metadata_path}")
    print(f"Failed image log saved to: {failed_path}")


if __name__ == "__main__":
    main()