from pathlib import Path

import pandas as pd

from app.core.config import settings


def resolve_existing_image_path(dataset_root: Path, relative_path: str) -> Path | None:
    """
    Try multiple possible DeepFashion layouts.
    Your mappings contain paths like:
    img/Sheer_Pleated-Front_Blouse/img_00000001.jpg
    """
    candidates = [
        dataset_root / relative_path,                 # if RAW_DATASET_ROOT = .../Img
        dataset_root / "img" / relative_path,         # if RAW_DATASET_ROOT = .../DeepFashion
        dataset_root.parent / "Img" / relative_path,  # fallback
    ]

    for p in candidates:
        if p.exists():
            return p
    return None


def parse_coarse_categories(category_file: Path) -> dict[int, str]:
    """
    Parse:
    50
    category_name  category_type
    Anorak         1
    Blazer         1
    ...
    """
    coarse_label_to_name = {}

    with open(category_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    # Skip first 2 lines: count + header
    data_lines = lines[2:]

    # We assign sequential labels 1..N based on file order
    # because DeepFashion list_category_cloth.txt is ordered categories.
    # The "category_type" column is NOT the unique label, it's garment group type.
    # The actual image mapping file uses 1..50 labels by row order.
    for idx, line in enumerate(data_lines, start=1):
        # split on multiple spaces/tabs robustly
        parts = line.split()
        if len(parts) < 2:
            continue

        # last token is category_type (1/2/3), NOT label
        # category name is everything before that
        coarse_name = " ".join(parts[:-1]).strip()

        coarse_label_to_name[idx] = coarse_name

    return coarse_label_to_name


def main():
    dataset_root = settings.resolve_path(settings.RAW_DATASET_ROOT)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # If RAW_DATASET_ROOT is .../DeepFashion/Img
    # then CategoryAttribute is sibling of Img
    # If RAW_DATASET_ROOT is .../DeepFashion
    # then CategoryAttribute is child
    possible_category_roots = [
        dataset_root.parent / "CategoryAttribute",
        dataset_root / "CategoryAttribute",
    ]

    category_attr_root = None
    for p in possible_category_roots:
        if p.exists():
            category_attr_root = p
            break

    if category_attr_root is None:
        raise FileNotFoundError(
            f"Could not find CategoryAttribute folder near dataset root: {dataset_root}"
        )

    anno_coarse_dir = category_attr_root / "AnnoCoarse"
    category_file = anno_coarse_dir / "list_category_cloth.txt"
    mapping_file = anno_coarse_dir / "list_category_img.txt"

    if not category_file.exists():
        raise FileNotFoundError(f"Category definition file not found: {category_file}")
    if not mapping_file.exists():
        raise FileNotFoundError(f"Image-category mapping file not found: {mapping_file}")

    print(f"[INFO] Reading category definitions from: {category_file}")
    coarse_label_to_name = parse_coarse_categories(category_file)
    print(f"[INFO] Loaded {len(coarse_label_to_name)} coarse categories")

    print(f"[INFO] Reading image mappings from: {mapping_file}")

    rows = []
    missing_files = 0

    with open(mapping_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    total_rows = int(lines[0])
    print(f"[INFO] Loaded {total_rows} image rows")

    image_lines = lines[2:]  # skip count + header

    for idx, line in enumerate(image_lines, start=1):
        parts = line.split()
        if len(parts) < 2:
            continue

        relative_path = parts[0].replace("\\", "/")
        coarse_label = int(parts[-1])

        abs_path = resolve_existing_image_path(dataset_root, relative_path)
        if abs_path is None:
            missing_files += 1
            continue

        image_id = Path(relative_path).stem

        # Fine-grained folder category
        path_parts = Path(relative_path).parts
        category = path_parts[-2] if len(path_parts) >= 2 else "unknown"

        title = category.replace("_", " ")

        rows.append({
            "image_id": image_id,
            "image_path": relative_path,
            "category": category,
            "title": title,
            "coarse_label": coarse_label,
            "coarse_name": coarse_label_to_name.get(coarse_label, f"Class_{coarse_label}"),
        })

        if idx % 50000 == 0:
            print(f"[INFO] Processed {idx}/{total_rows} rows...")

    if not rows:
        raise RuntimeError(
            "No rows were written. Image path resolution failed. "
            f"Dataset root currently resolves to: {dataset_root}"
        )

    metadata_df = pd.DataFrame(rows)

    output_path = settings.resolve_path(settings.METADATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False)

    print("\n=== METADATA GENERATION COMPLETE ===")
    print(f"Saved metadata to: {output_path}")
    print(f"Rows written: {len(metadata_df)}")
    print(f"Missing image files skipped: {missing_files}")
    print(f"Unique fine-grained folder categories: {metadata_df['category'].nunique()}")
    print(f"Unique coarse labels: {metadata_df['coarse_label'].nunique()}")
    print(f"Unique coarse names: {metadata_df['coarse_name'].nunique()}")


if __name__ == "__main__":
    main()