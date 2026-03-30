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
FULL_METADATA_PATH = "data/metadata.csv"

# Output paths for FULL build
FULL_EMBEDDINGS_PATH = "embeddings/image_embeddings_full.npy"
FULL_EMBEDDINGS_FP16_PATH = "embeddings/image_embeddings_full_fp16.npy"
FULL_EMBEDDING_MAP_PATH = "embeddings/embedding_map_full.csv"
FULL_GLOBAL_INDEX_PATH = "indexes/global_full.index"
FULL_INDEX_META_PATH = "indexes/index_meta_full.json"

# Coarse-group indexes (based on coarse_label = 1 / 2 / 3)
BUILD_COARSE_INDEXES = True
COARSE_INDEX_DIR = "indexes/coarse_indexes_full"


def load_image_safe(image_path: Path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def ensure_directories():
    settings.resolve_path(settings.EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
    settings.resolve_path(settings.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    if BUILD_COARSE_INDEXES:
        settings.resolve_path(COARSE_INDEX_DIR).mkdir(parents=True, exist_ok=True)


def load_metadata() -> pd.DataFrame:
    metadata_path = settings.resolve_path(FULL_METADATA_PATH)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Full metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required_cols = {"image_id", "image_path", "category", "coarse_label"}
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

    print(f"[INFO] Starting FULL embedding generation for {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding FULL dataset"):
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

            if len(valid_rows) % 10000 == 0:
                print(f"[INFO] Embedded {len(valid_rows)} images so far...")

    if batch_images:
        batch_embeddings = clip_service.embed_image_batch(batch_images)
        all_embeddings.append(batch_embeddings)
        valid_rows.extend(batch_rows)

    if not all_embeddings:
        raise RuntimeError("No embeddings were generated. Check metadata and dataset paths.")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    failed_df = pd.DataFrame(failed_rows)

    return embeddings, valid_df, failed_df


def save_embeddings(embeddings: np.ndarray, valid_df: pd.DataFrame):
    embeddings_path = settings.resolve_path(FULL_EMBEDDINGS_PATH)
    np.save(embeddings_path, embeddings)
    print(f"[INFO] Saved float32 embeddings: {embeddings_path}")

    if SAVE_FP16_COPY:
        embeddings_fp16 = embeddings.astype(np.float16)
        fp16_path = settings.resolve_path(FULL_EMBEDDINGS_FP16_PATH)
        np.save(fp16_path, embeddings_fp16)
        print(f"[INFO] Saved float16 embeddings: {fp16_path}")

    keep_cols = ["image_id", "image_path", "category", "coarse_label"]
    if "title" in valid_df.columns:
        keep_cols.append("title")

    embedding_map = valid_df[keep_cols].copy()

    embedding_map_path = settings.resolve_path(FULL_EMBEDDING_MAP_PATH)
    embedding_map.to_csv(embedding_map_path, index=False)
    print(f"[INFO] Saved embedding map: {embedding_map_path}")

    failed_path = settings.resolve_path("data/failed_images_full.csv")
    return failed_path


def build_global_index(embeddings: np.ndarray):
    print("[INFO] Building FULL global FAISS index...")
    index = FAISSService.build_index(embeddings)
    global_index_path = settings.resolve_path(FULL_GLOBAL_INDEX_PATH)
    FAISSService.save_index(index, global_index_path)
    print(f"[INFO] Saved FULL global index: {global_index_path}")
    return index


def build_coarse_indexes(embeddings: np.ndarray, valid_df: pd.DataFrame):
    if not BUILD_COARSE_INDEXES:
        return []

    print("[INFO] Building coarse-group FAISS indexes (coarse_label = 1/2/3)...")
    built_labels = []

    grouped = valid_df.groupby("coarse_label")

    for coarse_label, group_df in grouped:
        indices = group_df.index.to_list()
        coarse_embeddings = embeddings[indices]

        if len(coarse_embeddings) == 0:
            continue

        coarse_index = FAISSService.build_index(coarse_embeddings)
        coarse_index_path = settings.resolve_path(COARSE_INDEX_DIR) / f"coarse_{int(coarse_label)}.index"
        FAISSService.save_index(coarse_index, coarse_index_path)
        built_labels.append(int(coarse_label))

        print(f"[INFO] Built coarse index for label={int(coarse_label)} with {len(group_df)} images")

    return built_labels


def save_index_metadata(embeddings: np.ndarray, valid_df: pd.DataFrame, built_coarse_labels: List[int]):
    coarse_counts = (
        valid_df["coarse_label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    meta = {
        "embedding_dim": int(embeddings.shape[1]),
        "dataset_size": int(len(valid_df)),
        "index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "unique_product_categories": int(valid_df["category"].nunique()),
        "coarse_labels_present": sorted([int(x) for x in valid_df["coarse_label"].unique().tolist()]),
        "coarse_label_counts": {str(int(k)): int(v) for k, v in coarse_counts.items()},
        "coarse_indexes_built": built_coarse_labels,
    }

    index_meta_path = settings.resolve_path(FULL_INDEX_META_PATH)
    with open(index_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved FULL index metadata: {index_meta_path}")


def main():
    ensure_directories()

    print("[INFO] Loading CLIP model...")
    clip_service = CLIPService()

    print("[INFO] Loading FULL metadata...")
    df = load_metadata()

    embeddings, valid_df, failed_df = build_embeddings(df, clip_service)

    failed_path = save_embeddings(embeddings, valid_df)

    if not failed_df.empty:
        failed_df.to_csv(failed_path, index=False)
        print(f"[INFO] Saved failed image log: {failed_path}")

    build_global_index(embeddings)
    built_coarse_labels = build_coarse_indexes(embeddings, valid_df)

    save_index_metadata(embeddings, valid_df, built_coarse_labels)

    print("\n=== FULL BUILD COMPLETE ===")
    print(f"Final indexed images: {len(valid_df)}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Unique product-folder categories: {valid_df['category'].nunique()}")
    print(f"Coarse labels present: {sorted(valid_df['coarse_label'].unique().tolist())}")
    print(f"Global index: {settings.resolve_path(FULL_GLOBAL_INDEX_PATH)}")


if __name__ == "__main__":
    main()