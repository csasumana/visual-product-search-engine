import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.core.clip_model import CLIPService
from app.core.config import settings
from app.core.faiss_index import FAISSService
from app.core.metrics import MetricsTracker
from app.core.utils import read_upload_as_pil
from app.schemas.response import HealthResponse, MetricsResponse, SearchResponse

clip_service: Optional[CLIPService] = None
faiss_service: Optional[FAISSService] = None
metrics_tracker = MetricsTracker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_service, faiss_service

    try:
        print("[INFO] Initializing CLIP service...")
        clip_service = CLIPService()
        print("[INFO] CLIP service initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize CLIP service: {e}")
        clip_service = None

    try:
        print("[INFO] Initializing FAISS service...")
        faiss_service = FAISSService()
        faiss_service.initialize()
        print("[INFO] FAISS service initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize FAISS service: {e}")
        faiss_service = None

    yield


app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "app_name": settings.APP_NAME,
        "message": "Visual Product Search API is running.",
        "docs": "/docs",
        "health": "/health",
        "search_image": "/search/image",
        "search_text": "/search/text",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    clip_loaded = clip_service is not None
    index_loaded = faiss_service is not None and faiss_service.is_ready()

    dataset_size = faiss_service.get_dataset_size() if faiss_service else 0
    embedding_dim = clip_service.embedding_dim if clip_service else 0
    device = clip_service.device if clip_service else "unavailable"
    available_coarse_labels = faiss_service.get_available_coarse_labels() if faiss_service else []

    return HealthResponse(
        app_name=settings.APP_NAME,
        status="ok" if clip_loaded and index_loaded else "degraded",
        clip_loaded=clip_loaded,
        index_loaded=index_loaded,
        dataset_size=dataset_size,
        embedding_dim=embedding_dim,
        device=device,
        available_coarse_labels=available_coarse_labels,
    )


@app.post("/search/image", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=settings.DEFAULT_TOP_K),
    coarse_label: Optional[int] = Form(default=None),
):
    if clip_service is None:
        raise HTTPException(status_code=503, detail="CLIP model is not loaded.")

    if faiss_service is None or not faiss_service.is_ready():
        raise HTTPException(status_code=503, detail="FAISS index is not loaded.")

    image = await read_upload_as_pil(file)

    start = metrics_tracker.start_timer()
    query_embedding = clip_service.embed_image(image)
    results = faiss_service.search(
        query_embedding=query_embedding,
        top_k=top_k,
        coarse_label=coarse_label,
    )
    latency_ms = metrics_tracker.stop_timer(start)

    return SearchResponse(
        query_type="image",
        top_k=min(top_k, settings.MAX_SEARCH_K),
        coarse_label_filter=coarse_label,
        latency_ms=round(latency_ms, 3),
        results=results,
    )


@app.post("/search/text", response_model=SearchResponse)
async def search_text(
    text: str = Form(...),
    top_k: int = Form(default=settings.DEFAULT_TOP_K),
    coarse_label: Optional[int] = Form(default=None),
):
    if clip_service is None:
        raise HTTPException(status_code=503, detail="CLIP model is not loaded.")

    if faiss_service is None or not faiss_service.is_ready():
        raise HTTPException(status_code=503, detail="FAISS index is not loaded.")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text query cannot be empty.")

    start = metrics_tracker.start_timer()
    query_embedding = clip_service.embed_text(text)
    results = faiss_service.search(
        query_embedding=query_embedding,
        top_k=top_k,
        coarse_label=coarse_label,
    )
    latency_ms = metrics_tracker.stop_timer(start)

    return SearchResponse(
        query_type="text",
        top_k=min(top_k, settings.MAX_SEARCH_K),
        coarse_label_filter=coarse_label,
        latency_ms=round(latency_ms, 3),
        results=results,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    index_loaded = faiss_service is not None and faiss_service.is_ready()
    dataset_size = faiss_service.get_dataset_size() if faiss_service else 0
    embedding_dim = clip_service.embedding_dim if clip_service else 0

    summary = metrics_tracker.summary()

    return MetricsResponse(
        dataset_size=dataset_size,
        index_loaded=index_loaded,
        embedding_dim=embedding_dim,
        total_requests=summary["total_requests"],
        last_latency_ms=summary["last_latency_ms"],
        avg_latency_ms=summary["avg_latency_ms"],
        min_latency_ms=summary["min_latency_ms"],
        max_latency_ms=summary["max_latency_ms"],
    )