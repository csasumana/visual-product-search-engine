from typing import List, Optional

from pydantic import BaseModel


class SearchResult(BaseModel):
    image_id: str
    image_path: str
    category: str
    coarse_label: Optional[int] = None
    coarse_name: Optional[str] = None
    title: Optional[str] = None
    score: float


class SearchResponse(BaseModel):
    query_type: str
    top_k: int
    category_filter: Optional[str] = None
    latency_ms: float
    results: List[SearchResult]


class HealthResponse(BaseModel):
    app_name: str
    status: str
    clip_loaded: bool
    index_loaded: bool
    dataset_size: int
    embedding_dim: int
    device: str
    available_coarse_labels: List[int] = []
    available_coarse_names: List[str] = []


class MetricsResponse(BaseModel):
    dataset_size: int
    index_loaded: bool
    embedding_dim: int
    total_requests: int
    last_latency_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float