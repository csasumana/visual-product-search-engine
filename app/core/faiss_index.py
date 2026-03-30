from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd

from app.core.config import settings


# FINAL FULL-BUILD ARTIFACTS
FULL_EMBEDDING_MAP_PATH = "embeddings/embedding_map_full.csv"
FULL_GLOBAL_INDEX_PATH = "indexes/global_full.index"
FULL_COARSE_INDEX_DIR = "indexes/coarse_indexes_full"


class FAISSService:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata_df: Optional[pd.DataFrame] = None

        # coarse_label -> faiss.Index
        self.coarse_indexes: Dict[int, faiss.Index] = {}

        # coarse_label -> pd.DataFrame (aligned row order for that index)
        self.coarse_metadata: Dict[int, pd.DataFrame] = {}

    def load_metadata(self) -> None:
        """
        IMPORTANT:
        Use embedding_map_full.csv (not metadata.csv) because this file is guaranteed
        to align exactly with the final embedding row order and FAISS index row order.
        """
        metadata_path = settings.resolve_path(FULL_EMBEDDING_MAP_PATH)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Embedding map file not found: {metadata_path}")

        self.metadata_df = pd.read_csv(metadata_path)

        required_cols = {"image_id", "image_path", "category", "coarse_label"}
        missing = required_cols - set(self.metadata_df.columns)
        if missing:
            raise ValueError(f"embedding_map_full.csv missing required columns: {missing}")

        self.metadata_df = self.metadata_df.reset_index(drop=True)

    def load_global_index(self) -> None:
        index_path = settings.resolve_path(FULL_GLOBAL_INDEX_PATH)
        if not index_path.exists():
            raise FileNotFoundError(f"Full global FAISS index not found: {index_path}")

        self.index = faiss.read_index(str(index_path))

    def load_coarse_indexes(self) -> None:
        """
        Loads indexes like:
        indexes/coarse_indexes_full/coarse_1.index
        indexes/coarse_indexes_full/coarse_18.index
        ...
        """
        coarse_dir = settings.resolve_path(FULL_COARSE_INDEX_DIR)
        if not coarse_dir.exists():
            return

        if self.metadata_df is None:
            raise RuntimeError("Metadata must be loaded before coarse indexes.")

        for index_file in coarse_dir.glob("coarse_*.index"):
            stem = index_file.stem  # e.g., coarse_18
            try:
                coarse_label = int(stem.split("_")[1])
            except (IndexError, ValueError):
                continue

            self.coarse_indexes[coarse_label] = faiss.read_index(str(index_file))

            coarse_df = (
                self.metadata_df[self.metadata_df["coarse_label"] == coarse_label]
                .reset_index(drop=True)
            )
            self.coarse_metadata[coarse_label] = coarse_df

    def initialize(self) -> None:
        self.load_metadata()
        self.load_global_index()
        self.load_coarse_indexes()

    def is_ready(self) -> bool:
        return self.index is not None and self.metadata_df is not None

    def get_dataset_size(self) -> int:
        if self.metadata_df is None:
            return 0
        return len(self.metadata_df)

    def get_available_coarse_labels(self) -> List[int]:
        return sorted(self.coarse_indexes.keys())

    def _format_results(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
        metadata_df: pd.DataFrame
    ) -> List[dict]:
        results = []

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(metadata_df):
                continue

            row = metadata_df.iloc[int(idx)]

            result = {
                "image_id": str(row["image_id"]),
                "image_path": str(row["image_path"]),
                "category": str(row["category"]),
                "title": str(row["title"]) if "title" in metadata_df.columns and pd.notna(row["title"]) else None,
                "coarse_label": int(row["coarse_label"]) if pd.notna(row["coarse_label"]) else None,
                "score": float(score),
            }
            results.append(result)

        return results

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        coarse_label: Optional[int] = None,
    ) -> List[dict]:
        if not self.is_ready():
            raise RuntimeError("FAISS service is not initialized.")

        top_k = min(top_k, settings.MAX_SEARCH_K)
        query_embedding = query_embedding.astype(np.float32)

        # Category-restricted search using prebuilt coarse index
        if coarse_label is not None and coarse_label in self.coarse_indexes:
            coarse_index = self.coarse_indexes[coarse_label]
            coarse_meta = self.coarse_metadata.get(coarse_label)

            if coarse_meta is None or coarse_meta.empty:
                return []

            scores, indices = coarse_index.search(query_embedding, top_k)
            return self._format_results(indices, scores, coarse_meta)

        # Default: global search
        scores, indices = self.index.search(query_embedding, max(top_k * 3, top_k))
        results = self._format_results(indices, scores, self.metadata_df)

        # Optional fallback filter if caller passed a coarse label that wasn't preloaded
        if coarse_label is not None:
            results = [r for r in results if r["coarse_label"] == coarse_label]

        return results[:top_k]

    @staticmethod
    def build_index(embeddings: np.ndarray) -> faiss.Index:
        """
        Build cosine-similarity-compatible FAISS index.
        IMPORTANT:
        CLIP embeddings should already be L2-normalized before being added.
        Then inner product == cosine similarity.
        """
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    @staticmethod
    def save_index(index: faiss.Index, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))