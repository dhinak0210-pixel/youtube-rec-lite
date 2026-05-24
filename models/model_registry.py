"""
Central Model Registry and Coordinator.

Integrates and orchestrates the lifecycle of all retrieval models 
and the MMoE neural ranker. Provides unified interfaces for bulk model 
fitting, multi-candidate retrieval, and multitask expert ranking.
Supports fitting using both Pydantic lists and legacy CSV tables.
"""

import os
from typing import List as TypeList, Optional
from loguru import logger
from models.collaborative_filtering import CollaborativeFilteringRecommender
from models.matrix_factorization import PyTorchMatrixFactorization
from models.sequential_recommender import PyTorchSequentialRecommender
from models.graph_neural_network import PyTorchGNNRecommender
from models.multi_objective_ranker import PyTorchMMoERanker
from models.cold_start import ColdStartRecommender

class ModelRegistry:
    """Orchestrates candidate generation retrieval pipelines and MMoE re-ranking."""

    def __init__(self):
        """Initializes all retrieval engines, neural modules, and fallbacks."""
        self.cf = CollaborativeFilteringRecommender()
        self.mf = PyTorchMatrixFactorization()
        self.seq = PyTorchSequentialRecommender()
        self.gnn = PyTorchGNNRecommender()
        self.mmoe = PyTorchMMoERanker()
        self.cold = ColdStartRecommender()
        self.is_fitted = False

    def fit_all(self, users=None, items=None, interactions=None, data_dir: str = "data"):
        """Trains all retrieval models and MMoE multi-objective networks sequentially."""
        logger.info("🚀 Starting bulk training of all RecoStream AI models...")
        
        # 1. Cold Start Heuristics
        self.cold.fit(users=users, items=items, interactions=interactions, data_dir=data_dir)
        
        # 2. Memory-based Item Similarity
        self.cf.fit(users=users, items=items, interactions=interactions, data_dir=data_dir)
        
        # 3. Latent Embeddings Matrix Factorization (PyTorch)
        self.mf.fit(users=users, items=items, interactions=interactions, data_dir=data_dir)
        
        # 4. Attention-based Transformer sequence learning (PyTorch)
        self.seq.fit(users=users, items=items, interactions=interactions, data_dir=data_dir)
        
        # 5. Neighborhood propagation GNN (PyTorch)
        self.gnn.fit(users=users, items=items, interactions=interactions, data_dir=data_dir)
        
        # 6. Multi-task gating MMoE ranker (PyTorch)
        self.mmoe.fit(users=users, items=items, interactions=interactions, data_dir=data_dir)
        
        self.is_fitted = True
        logger.info("🎉 Bulk training of all AI modules completed successfully!")

    def retrieve_candidates(self, user_id: int, source: str, pool_size: int = 100) -> list:
        """Retrieves raw candidates from a specified retrieval pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Model registry must be fitted before scoring.")
            
        # Determine if the user has no history (Cold Start fallback)
        if self.cold.is_cold_user(user_id):
            return self.cold.recommend(user_id, top_n=pool_size)
            
        source = source.upper()
        if source == "CF":
            candidates = self.cf.recommend(user_id, top_n=pool_size)
        elif source == "MF":
            candidates = self.mf.recommend(user_id, top_n=pool_size)
        elif source in ("BERT4REC", "SEQ"):
            candidates = self.seq.recommend(user_id, top_n=pool_size)
        elif source == "GNN":
            candidates = self.gnn.recommend(user_id, top_n=pool_size)
        elif source == "COLD":
            candidates = self.cold.recommend(user_id, top_n=pool_size)
        else:
            # Union/Hybrid default candidate generation
            cf_c = self.cf.recommend(user_id, top_n=pool_size // 2)
            mf_c = self.mf.recommend(user_id, top_n=pool_size // 2)
            candidates = list(set(cf_c + mf_c))
            
        return candidates

    def rank_candidates(self, user_id: int, candidates: list) -> list:
        """Re-scores raw candidate lists using the Multi-gate Mixture-of-Experts neural ranker."""
        if not self.is_fitted:
            raise RuntimeError("Model registry must be fitted before ranking.")
            
        # Extract item IDs
        vids = [c[0] for c in candidates]
        return self.mmoe.rank(user_id, vids)

# Instantiate a registry singleton to act as a central reference
model_registry = ModelRegistry()
