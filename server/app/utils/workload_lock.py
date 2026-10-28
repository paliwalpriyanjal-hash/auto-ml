# server/app/utils/workload_lock.py
"""
Lightweight concurrency limiter for heavy compute workloads (Optuna tuning, model comparison).
Prevents CPU starvation on single-node / laptop deployments without requiring external Redis/Celery.
"""

import os
import threading
from contextlib import asynccontextmanager
from fastapi import HTTPException


class HeavyWorkloadLimiter:
    """Bounded semaphore limiting concurrent CPU-intensive AutoML tasks."""

    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_jobs = 0
        self._lock = threading.Lock()

    @asynccontextmanager
    async def acquire_slot(self, task_name: str = "AutoML workload", timeout: float = 1.0):
        acquired = self._semaphore.acquire(blocking=True, timeout=timeout)
        if not acquired:
            raise HTTPException(
                status_code=429,
                detail=f"Server is currently executing another heavy {task_name}. Please wait for it to finish.",
            )
        with self._lock:
            self._active_jobs += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_jobs = max(0, self._active_jobs - 1)
            self._semaphore.release()

    @property
    def active_jobs_count(self) -> int:
        with self._lock:
            return self._active_jobs


max_concurrent = int(os.getenv("AUTOVEYRA_MAX_CONCURRENT_TRAININGS", "1"))
workload_limiter = HeavyWorkloadLimiter(max_concurrent=max_concurrent)
