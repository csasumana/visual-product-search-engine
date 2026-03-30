import time
from collections import deque
from typing import Deque, Dict


class MetricsTracker:
    def __init__(self, max_history: int = 200):
        self.latencies_ms: Deque[float] = deque(maxlen=max_history)
        self.last_latency_ms: float = 0.0
        self.total_requests: int = 0

    def start_timer(self) -> float:
        return time.perf_counter()

    def stop_timer(self, start_time: float) -> float:
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.last_latency_ms = latency_ms
        self.latencies_ms.append(latency_ms)
        self.total_requests += 1
        return latency_ms

    def summary(self) -> Dict:
        if not self.latencies_ms:
            return {
                "total_requests": self.total_requests,
                "last_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
            }

        latencies = list(self.latencies_ms)
        return {
            "total_requests": self.total_requests,
            "last_latency_ms": round(self.last_latency_ms, 3),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
            "min_latency_ms": round(min(latencies), 3),
            "max_latency_ms": round(max(latencies), 3),
        }