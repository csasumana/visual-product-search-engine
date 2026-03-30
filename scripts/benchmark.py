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
    sample_size = 100
    random_seed = 42

    print("[INFO] Initializing CLIP service...")
    clip_service = CLIPService()

    print("[INFO] Initializing FAISS service...")
    faiss_service = FAISSService()
    faiss_service.initialize()

    if faiss_service.metadata_df is None:
        raise RuntimeError("Metadata not loaded.")

    metadata = faiss_service.metadata_df.copy()
    sample_df = metadata.sample(n=min(sample_size, len(metadata)), random_state=random_seed).reset_index(drop=True)

    embed_latencies = []
    search_latencies = []
    total_latencies = []

    missing_paths = 0
    open_failures = 0

    for i, row in sample_df.iterrows():
        rel_path = str(row["image_path"])
        image_path = resolve_dataset_image_path(rel_path)

        if not image_path.exists():
            missing_paths += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            open_failures += 1
            continue

        t0 = time.perf_counter()
        embedding = clip_service.embed_image(image)
        t1 = time.perf_counter()

        _ = faiss_service.search(query_embedding=embedding, top_k=5)
        t2 = time.perf_counter()

        embed_ms = (t1 - t0) * 1000.0
        search_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0

        embed_latencies.append(embed_ms)
        search_latencies.append(search_ms)
        total_latencies.append(total_ms)

        if (i + 1) % 20 == 0:
            print(f"[INFO] Benchmarked {i+1}/{len(sample_df)} queries...")

    print("\n=== DEBUG SUMMARY ===")
    print(f"Missing paths: {missing_paths}")
    print(f"Open failures: {open_failures}")

    if not total_latencies:
        raise RuntimeError("No valid benchmark samples processed.")

    metrics = {
        "queries_benchmarked": len(total_latencies),
        "avg_embed_latency_ms": round(mean(embed_latencies), 3),
        "p95_embed_latency_ms": round(percentile(embed_latencies, 95), 3),
        "avg_search_latency_ms": round(mean(search_latencies), 3),
        "p95_search_latency_ms": round(percentile(search_latencies, 95), 3),
        "avg_total_latency_ms": round(mean(total_latencies), 3),
        "p95_total_latency_ms": round(percentile(total_latencies, 95), 3),
        "min_total_latency_ms": round(min(total_latencies), 3),
        "max_total_latency_ms": round(max(total_latencies), 3),
    }

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = artifacts_dir / "benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== BENCHMARK SUMMARY ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\n[INFO] Saved benchmark metrics to: {metrics_path}")


if __name__ == "__main__":
    main()