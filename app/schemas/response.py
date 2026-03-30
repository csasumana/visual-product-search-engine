from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class SearchResult(BaseModel):
    image_id: str
    image_path: str
    category: str
    title: Optional[str] = None
    coarse_label: Optional[int] = None
    score: float


class SearchResponse(BaseModel):
    query_type: str
    top_k: int
    coarse_label_filter: Optional[int] = None
    latency_ms: float
    results: List[SearchResult]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    app_name: str
    status: str
    clip_loaded: bool
    index_loaded: bool
    dataset_size: int
    embedding_dim: int
    device: str
    available_coarse_labels: List[int]


class MetricsResponse(BaseModel):
    dataset_size: int
    index_loaded: bool
    embedding_dim: int
    total_requests: int
    last_latency_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float