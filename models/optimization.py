"""
Model Optimization and Inference Acceleration Engine.

Implements:
1. PyTorch Dynamic Model Quantization (float32 -> int8 conversions).
2. LRU Embedding and Recommendation Cache (fast lookups bypassing neural passes).
3. Real-Time Performance Profiler and Benchmarking harness.
"""

import time
import sys
import copy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

class DynamicQuantizer:
    """
    Applies Post-Training Dynamic Quantization to PyTorch recommendation modules
    (such as sequential Transformers, GNN Layers, or MMoE Tower networks).
    Converts linear model weights from float32 to int8 to reduce memory footprint and latency.
    """
    @staticmethod
    def quantize(model: nn.Module, qdtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Dynamically quantizes all linear/projection layers within the provided PyTorch model.
        """
        logger.info(f"Applying post-training dynamic quantization to model: {model.__class__.__name__}")
        try:
            # Dynamic quantization targets Linear layers
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LayerNorm},
                dtype=qdtype
            )
            logger.info("Dynamic quantization applied successfully!")
            return quantized_model
        except Exception as e:
            logger.warning(f"Failed to dynamically quantize model. Fallback to raw copy. Error: {e}")
            return copy.deepcopy(model)

    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """
        Calculates the memory footprint of the PyTorch model state dict in megabytes.
        """
        temp_path = "temp_model_size.pt"
        try:
            torch.save(model.state_dict(), temp_path)
            size_bytes = sys.getsizeof(open(temp_path, 'rb').read())
            import os
            os.remove(temp_path)
            return float(size_bytes) / (1024 * 1024)
        except Exception:
            # Fallback estimation using parameter counts
            param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
            return float(param_size + buffer_size) / (1024 * 1024)

class LRUEmbeddingCache:
    """
    High-performance Least Recently Used (LRU) inference cache.
    Caches calculated user context embeddings, item scores, or candidate recommendation
    lists to bypass heavy GPU/CPU forward passes during peak high-QPS traffic.
    """
    def __init__(self, capacity: int = 1000, ttl_seconds: float = 300.0):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache: Dict[Any, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self.access_order: List[Any] = []  # Tracks access recency

    def get(self, key: Any) -> Optional[Any]:
        """Retrieves cached value if it exists, is not expired, and re-orders access priority."""
        if key not in self.cache:
            return None

        val, ts = self.cache[key]
        if time.time() - ts > self.ttl:
            # Expired cache element
            self.delete(key)
            return None

        # Re-prioritize to end of access list (recently used)
        self.access_order.remove(key)
        self.access_order.append(key)
        return val

    def put(self, key: Any, value: Any):
        """Inserts key-value pair and evicts least recently used items if capacity is exceeded."""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Evict least recently used item (first element in access_order)
            lru_key = self.access_order.pop(0)
            self.cache.pop(lru_key, None)

        self.cache[key] = (value, time.time())
        self.access_order.append(key)

    def delete(self, key: Any):
        """Explicitly deletes a key from cache."""
        self.cache.pop(key, None)
        if key in self.access_order:
            self.access_order.remove(key)

    def clear(self):
        """Clears all cache elements."""
        self.cache.clear()
        self.access_order.clear()

class PerformanceProfiler:
    """
    Comprehensive profiler and benchmarking suite to measure and analyze recommendation latency,
    throughput, model sizes, and memory usage.
    """
    @staticmethod
    def benchmark_inference(model: nn.Module, sample_input: Any, num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmarks inference latencies and throughput of a PyTorch module.
        """
        # Warmup passes
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                if isinstance(sample_input, tuple):
                    _ = model(*sample_input)
                elif isinstance(sample_input, dict):
                    _ = model(**sample_input)
                else:
                    _ = model(sample_input)

        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                if isinstance(sample_input, tuple):
                    _ = model(*sample_input)
                elif isinstance(sample_input, dict):
                    _ = model(**sample_input)
                else:
                    _ = model(sample_input)
                latencies.append(time.perf_counter() - start)

        latencies_ms = np.array(latencies) * 1000.0
        return {
            "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
            "avg_latency_ms": float(np.mean(latencies_ms)),
            "std_latency_ms": float(np.std(latencies_ms)),
            "qps_throughput": float(1.0 / max(np.mean(latencies), 1e-9))
        }

    @staticmethod
    def profile_recommendation_pipeline(registry: Any, user_id: int, num_queries: int = 50) -> Dict[str, Any]:
        """
        Measures recommendation latencies (in milliseconds) across individual retrieval channels
        and the final multi-objective re-ranking steps.
        """
        latencies = defaultdict(list)
        sources = ["CF", "MF", "SEQ", "GNN", "COLD", "HYBRID"]

        for _ in range(num_queries):
            # Benchmark Candidate Generations
            for src in sources:
                start = time.perf_counter()
                candidates = registry.retrieve_candidates(user_id=user_id, source=src, pool_size=100)
                latencies[src].append((time.perf_counter() - start) * 1000.0)

            # Benchmark Ranking Pipeline
            start = time.perf_counter()
            _ = registry.rank_candidates(user_id=user_id, candidates=candidates)
            latencies["MMOE_RANKING"].append((time.perf_counter() - start) * 1000.0)

        # Summarize averages
        summary = {}
        for key, val in latencies.items():
            summary[key] = {
                "avg_ms": float(np.mean(val)),
                "p95_ms": float(np.percentile(val, 95))
            }
        return summary
