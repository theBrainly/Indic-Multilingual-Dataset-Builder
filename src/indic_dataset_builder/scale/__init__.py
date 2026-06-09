"""Distributed-scale curation jobs (Dask / Spark-style).

Demonstrates the scaling path for the per-document stages on sharded Parquet:
exact dedup (shuffle-by-hash) and heuristic filtering (map over partitions).
Uses Dask when installed; otherwise runs the identical logic with pandas so the
behaviour is verifiable on a laptop and in CI.
"""
from .dask_jobs import distributed_dedup_filter, DEFAULT_BACKEND  # noqa: F401
