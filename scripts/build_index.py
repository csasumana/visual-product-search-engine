import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from app.core.clip_model import CLIPService
from app.core.config import settings
from app.core.faiss_index import FAISSService


BATCH_SIZE = 64
SAVE_FP16_COPY = True
BUILD_CATEGORY_INDEXES = True


def load_image_safe(image_path: Path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def ensure_directories():
    settings.resolve_path(settings.EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
    settings.resolve_path(settings.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    settings.resolve_path(settings.CATEGORY_INDEX_DIR).mkdir(parents=True, exist_ok=True)


def load_metadata() -> pd.DataFrame:
    metadata_path = settings.resolve_path(settings.METADATA_PATH)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required_cols = {"image_id", "image_path", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")

    df = df.reset_index(drop=True)
    return df


def build_embeddings(df: pd.DataFrame, clip_service: CLIPService):
    all_embeddings = []
    valid_rows = []
    failed_rows = []

    batch_images: List[Image.Image] = []
    batch_rows = []

    print(f"[INFO] Starting embedding generation for {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding images"):
        rel_path = str(row["image_path"])
        full_path = settings.resolve_raw_dataset_path(rel_path)

        image = load_image_safe(full_path)
        if image is None:
            failed_rows.append({
                "image_id": row["image_id"],
                "image_path": rel_path,
                "category": row["category"],
                "reason": "load_failed"
            })
            continue

        batch_images.append(image)
        batch_rows.append(row)

        if len(batch_images) == BATCH_SIZE:
            batch_embeddings = clip_service.embed_image_batch(batch_images)
            all_embeddings.append(batch_embeddings)
            valid_rows.extend(batch_rows)
            batch_images = []
            batch_rows = []

    if batch_images:
        batch_embeddings = clip_service.embed_image_batch(batch_images)
        all_embeddings.append(batch_embeddings)
        valid_rows.extend(batch_rows)

    if not all_embeddings:
        raise RuntimeError("No embeddings were generated. Check your dataset and metadata.")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    failed_df = pd.DataFrame(failed_rows)

    return embeddings, valid_df, failed_df


def save_embeddings(embeddings: np.ndarray, valid_df: pd.DataFrame):
    embeddings_path = settings.resolve_path(settings.EMBEDDINGS_PATH)
    np.save(embeddings_path, embeddings)
    print(f"[INFO] Saved float32 embeddings: {embeddings_path}")

    if SAVE_FP16_COPY:
        embeddings_fp16 = embeddings.astype(np.float16)
        fp16_path = settings.resolve_path(settings.EMBEDDINGS_FP16_PATH)
        np.save(fp16_path, embeddings_fp16)
        print(f"[INFO] Saved float16 embeddings: {fp16_path}")

    keep_cols = ["image_id", "image_path", "category"]
    if "title" in valid_df.columns:
        keep_cols.append("title")
    if "coarse_label" in valid_df.columns:
        keep_cols.append("coarse_label")

    embedding_map = valid_df[keep_cols].copy()

    embedding_map_path = settings.resolve_path(settings.EMBEDDING_MAP_PATH)
    embedding_map.to_csv(embedding_map_path, index=False)
    print(f"[INFO] Saved embedding map: {embedding_map_path}")

    metadata_path = settings.resolve_path(settings.METADATA_PATH)
    valid_df.to_csv(metadata_path, index=False)
    print(f"[INFO] Updated cleaned metadata: {metadata_path}")

    failed_path = settings.resolve_path("data/failed_images.csv")
    return failed_path


def build_global_index(embeddings: np.ndarray):
    print("[INFO] Building global FAISS index...")
    index = FAISSService.build_index(embeddings)
    global_index_path = settings.resolve_path(settings.GLOBAL_INDEX_PATH)
    FAISSService.save_index(index, global_index_path)
    print(f"[INFO] Saved global index: {global_index_path}")
    return index


def safe_category_filename(category: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if c in invalid_chars else c for c in str(category))
    return cleaned[:180]


def build_category_indexes(embeddings: np.ndarray, valid_df: pd.DataFrame):
    if not BUILD_CATEGORY_INDEXES:
        return []

    print("[INFO] Building category-specific FAISS indexes...")
    built_categories = []

    grouped = valid_df.groupby("category")
    for category, group_df in grouped:
        indices = group_df.index.to_list()
        category_embeddings = embeddings[indices]

        if len(category_embeddings) == 0:
            continue

        cat_index = FAISSService.build_index(category_embeddings)
        safe_name = safe_category_filename(category)
        cat_index_path = settings.resolve_path(settings.CATEGORY_INDEX_DIR) / f"{safe_name}.index"
        FAISSService.save_index(cat_index, cat_index_path)
        built_categories.append(category)

    print(f"[INFO] Built {len(built_categories)} category indexes.")
    return built_categories


def save_index_metadata(embeddings: np.ndarray, valid_df: pd.DataFrame, built_categories: List[str]):
    meta = {
        "embedding_dim": int(embeddings.shape[1]),
        "dataset_size": int(len(valid_df)),
        "index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "unique_categories": int(valid_df["category"].nunique()),
        "category_indexes_built": len(built_categories),
    }

    index_meta_path = settings.resolve_path(settings.INDEX_META_PATH)
    with open(index_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved index metadata: {index_meta_path}")


def main():
    ensure_directories()

    print("[INFO] Loading CLIP model...")
    clip_service = CLIPService()

    print("[INFO] Loading metadata...")
    df = load_metadata()

    embeddings, valid_df, failed_df = build_embeddings(df, clip_service)

    failed_path = save_embeddings(embeddings, valid_df)

    if not failed_df.empty:
        failed_df.to_csv(failed_path, index=False)
        print(f"[INFO] Saved failed image log: {failed_path}")

    build_global_index(embeddings)
    built_categories = build_category_indexes(embeddings, valid_df)

    save_index_metadata(embeddings, valid_df, built_categories)

    print("\n=== BUILD COMPLETE ===")
    print(f"Final indexed images: {len(valid_df)}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Unique categories: {valid_df['category'].nunique()}")
    print(f"Global index: {settings.resolve_path(settings.GLOBAL_INDEX_PATH)}")


if __name__ == "__main__":
    main()