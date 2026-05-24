"""
RecoStream Inference Latency and Performance Benchmarks.

Runs automated speed trials across all 5 candidate retrievers and the MMoE ranker.
Reports average execution times in milliseconds to prove low-latency production readiness.
"""

import time
import numpy as np
from models import model_registry

class RecoStreamBenchmarks:
    def __init__(self):
        self.registry = model_registry

    def run_all_benchmarks(self, num_trials: int = 50):
        """Runs latency trials across all retrieval pipelines and neural rankers."""
        print("🚀 Bootstrapping speed trials and performance benchmarks...")
        
        # Fit models if needed
        if not self.registry.is_fitted:
            self.registry.fit_all()
            
        sources = ["CF", "MF", "BERT4REC", "GNN", "COLD"]
        latencies = {s: [] for s in sources}
        latencies["MMOE_RANKING"] = []
        
        # Test User ID 42
        test_uid = 42
        
        print(f"⏱️ Running {num_trials} inference speed trials per pipeline...")
        for _ in range(num_trials):
            for src in sources:
                start = time.perf_counter()
                candidates = self.registry.retrieve_candidates(test_uid, src, pool_size=100)
                elapsed = (time.perf_counter() - start) * 1000.0 # ms
                latencies[src].append(elapsed)
                
                # Benchmark MMoE ranking speed
                if src == "CF" and candidates:
                    start_rank = time.perf_counter()
                    self.registry.rank_candidates(test_uid, candidates)
                    elapsed_rank = (time.perf_counter() - start_rank) * 1000.0 # ms
                    latencies["MMOE_RANKING"].append(elapsed_rank)
                    
        # Print latency summary reports
        print("\n================== RecoStream Speed Benchmark Report ==================")
        print(f"{'Retrieval Source / Pipeline':<35} | {'Mean Latency':<15} | {'p95 Latency':<12}")
        print("-" * 72)
        
        for k, vals in latencies.items():
            mean_lat = np.mean(vals)
            p95_lat = np.percentile(vals, 95)
            name = f"Retriever ({k})" if k in sources else "Ranker (MMoE)"
            print(f"{name:<35} | {mean_lat:>10.2f} ms | {p95_lat:>8.2f} ms")
            
        print("=======================================================================")
        print("✅ Speed benchmarks completed successfully!")

if __name__ == "__main__":
    bench = RecoStreamBenchmarks()
    bench.run_all_benchmarks(num_trials=20)
