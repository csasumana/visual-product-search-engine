from pathlib import Path
import pandas as pd

from app.core.config import settings


def load_image_category_rows(category_img_path: Path):
    """
    Reads list_category_img.txt
    Format:
    289222
    image_name  category_label
    img/Some_Product_Name/img_00000001.jpg    3
    ...
    """
    image_rows = []

    with open(category_img_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue

        image_path = parts[0].strip()
        coarse_label = parts[1].strip()

        if not image_path.lower().startswith("img/"):
            continue

        try:
            coarse_label = int(coarse_label)
        except ValueError:
            continue

        image_rows.append((image_path, coarse_label))

    return image_rows


def resolve_actual_image_path(raw_root: Path, relative_image_path: str) -> Path:
    """
    Your dataset layout:
    raw_root = F:/datasets/DeepFashion/CategoryAttribute
    annotation path = img/ProductFolder/img_00000001.jpg
    actual file     = F:/datasets/DeepFashion/CategoryAttribute/img/img/ProductFolder/img_00000001.jpg
    """
    return raw_root / "img" / relative_image_path


def extract_product_folder(relative_image_path: str) -> str:
    """
    Example:
    img/Sheer_Pleated-Front_Blouse/img_00000001.jpg
    -> Sheer_Pleated-Front_Blouse
    """
    parts = Path(relative_image_path).parts
    if len(parts) >= 3:
        return parts[1]
    return "unknown"


def clean_title(folder_name: str) -> str:
    """
    Example:
    Sheer_Pleated-Front_Blouse -> Sheer Pleated Front Blouse
    """
    return folder_name.replace("_", " ").replace("-", " ").strip()


def main():
    raw_root = Path(settings.RAW_DATASET_ROOT)
    if not raw_root.exists():
        raise FileNotFoundError(f"RAW_DATASET_ROOT does not exist: {raw_root}")

    anno_dir = raw_root / "AnnoCoarse"
    category_img_path = anno_dir / "list_category_img.txt"

    if not category_img_path.exists():
        raise FileNotFoundError(f"Missing file: {category_img_path}")

    print(f"[INFO] Reading image mappings from: {category_img_path}")
    image_rows = load_image_category_rows(category_img_path)
    print(f"[INFO] Loaded {len(image_rows)} image rows")

    metadata_rows = []
    missing_files = 0

    for idx, (relative_image_path, coarse_label) in enumerate(image_rows, start=1):
        full_path = resolve_actual_image_path(raw_root, relative_image_path)

        if not full_path.exists():
            missing_files += 1
            continue

        product_folder = extract_product_folder(relative_image_path)
        category = product_folder
        title = clean_title(product_folder)

        metadata_rows.append({
            "image_id": idx,
            "image_path": relative_image_path.replace("\\", "/"),
            "category": category,
            "title": title,
            "coarse_label": coarse_label,
        })

        if idx % 50000 == 0:
            print(f"[INFO] Processed {idx}/{len(image_rows)} rows...")

    metadata_df = pd.DataFrame(metadata_rows)

    output_path = settings.resolve_path(settings.METADATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False)

    print("\n=== METADATA GENERATION COMPLETE ===")
    print(f"Saved metadata to: {output_path}")
    print(f"Rows written: {len(metadata_df)}")
    print(f"Missing image files skipped: {missing_files}")
    print(f"Unique product-folder categories: {metadata_df['category'].nunique()}")


if __name__ == "__main__":
    main()