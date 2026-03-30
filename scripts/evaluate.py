import os
import time
from pathlib import Path
from statistics import mean
from typing import List
import json
import numpy as np

# Prevent OpenMP duplicate runtime crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image

from app.core.clip_model import CLIPService
from app.core.faiss_index import FAISSService
from app.core.config import settings


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def resolve_dataset_image_path(relative_path: str) -> Path:
    """
    metadata stores: img/ProductFolder/img_00000001.jpg
    actual file is: RAW_DATASET_ROOT/img/img/ProductFolder/img_00000001.jpg
    """
    if not settings.RAW_DATASET_ROOT:
        raise ValueError("RAW_DATASET_ROOT is not set in .env")

    return Path(settings.RAW_DATASET_ROOT) / "img" / relative_path


def main():
    sample_size = 200
    random_seed = 42
    max_k = 11  # request 11 so we can drop self-match and still evaluate @10

    print("[INFO] Initializing CLIP service...")
    clip_service = CLIPService()

    print("[INFO] Initializing FAISS service...")
    faiss_service = FAISSService()
    faiss_service.initialize()

    if faiss_service.metadata_df is None:
        raise RuntimeError("Metadata not loaded.")

    metadata = faiss_service.metadata_df.copy()

    required_cols = {"image_id", "image_path", "coarse_label"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise ValueError(f"Embedding map missing required columns: {missing}")

    total_rows = len(metadata)
    print(f"[INFO] Dataset size: {total_rows}")

    sample_df = metadata.sample(n=min(sample_size, total_rows), random_state=random_seed).reset_index(drop=True)

    p5_scores = []
    p10_scores = []
    latencies_ms = []

    missing_paths = 0
    open_failures = 0
    no_result_failures = 0

    for i, row in sample_df.iterrows():
        rel_path = str(row["image_path"])
        query_image_id = str(row["image_id"])
        query_coarse = int(row["coarse_label"])

        image_path = resolve_dataset_image_path(rel_path)

        if not image_path.exists():
            missing_paths += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            open_failures += 1
            continue

        query_embedding = clip_service.embed_image(image)

        start = time.perf_counter()
        raw_results = faiss_service.search(query_embedding=query_embedding, top_k=max_k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)

        if not raw_results:
            no_result_failures += 1
            continue

        # Remove exact self-match if present
        filtered = [r for r in raw_results if str(r["image_id"]) != query_image_id]

        if len(filtered) < 5:
            no_result_failures += 1
            continue

        top5 = filtered[:5]
        top10 = filtered[:10] if len(filtered) >= 10 else filtered

        hits5 = sum(
            1 for r in top5
            if r.get("coarse_label") is not None and int(r["coarse_label"]) == query_coarse
        )
        hits10 = sum(
            1 for r in top10
            if r.get("coarse_label") is not None and int(r["coarse_label"]) == query_coarse
        )

        p5_scores.append(hits5 / len(top5))
        p10_scores.append(hits10 / len(top10))

        if (i + 1) % 25 == 0:
            print(f"[INFO] Evaluated {i+1}/{len(sample_df)} queries...")

    print("\n=== DEBUG SUMMARY ===")
    print(f"Missing paths:      {missing_paths}")
    print(f"Open failures:      {open_failures}")
    print(f"No-result failures: {no_result_failures}")

    if not p5_scores:
        raise RuntimeError("No valid evaluation samples processed.")

    metrics = {
        "dataset_size": total_rows,
        "queries_evaluated": len(p5_scores),
        "precision_at_5": round(mean(p5_scores), 4),
        "precision_at_10": round(mean(p10_scores), 4),
        "avg_latency_ms": round(mean(latencies_ms), 3),
        "p95_latency_ms": round(percentile(latencies_ms, 95), 3),
        "min_latency_ms": round(min(latencies_ms), 3),
        "max_latency_ms": round(max(latencies_ms), 3),
    }

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = artifacts_dir / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== RETRIEVAL EVALUATION SUMMARY ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\n[INFO] Saved evaluation metrics to: {metrics_path}")

if __name__ == "__main__":
    main()