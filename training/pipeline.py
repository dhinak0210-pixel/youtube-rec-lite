"""
Model Training and Orchestration Pipeline.

Orchestrates the entire offline training lifecycle:
1. Synthetic pareto-curve data generation
2. User-Item Collaborative Filtering similarity fitting
3. Latent embedding ALS Matrix Factorization
4. Attention-based Transformer sequential prediction (BERT4Rec)
5. Neighborhood bipartite propagation GNN (GraphSAGE)
6. Popularity-based bandit Cold Start strategies
7. Multi-gate Mixture-of-Experts (MMoE) multi-task neural ranking
8. Accuracy evaluation using standard Information Retrieval (IR) metrics
9. Synchronization of all embeddings and historical features to Feature Store.
"""

import time
import numpy as np
from typing import List, Dict, Set, Optional, Any, Tuple
from collections import defaultdict
from loguru import logger

from config import settings
from data import DataGenerator, Interaction, UserProfile, ItemProfile
from models.collaborative_filtering import CollaborativeFilter
from models.matrix_factorization import MatrixFactorization
from models.sequential_recommender import BERT4RecTrainer
from models.graph_neural_network import GNNTrainer
from models.cold_start import ColdStartHandler
from models.multi_objective_ranker import MMoETrainer, FEATURE_DIM
from services.feature_store import FeatureStore
from evaluation.metrics import Metrics
from models import model_registry

class TrainingPipeline:
    """
    Unified RecoStream AI model training orchestration pipeline.
    
    Manages data generation, model fitting, metric evaluation, and feature store caching.
    """

    def __init__(self, num_users: int = settings.num_users, 
                 num_items: int = settings.num_items,
                 num_interactions: int = settings.num_interactions):
        """Initializes pipeline state configurations and model attributes."""
        self.num_users = num_users
        self.num_items = num_items
        self.num_interactions = num_interactions
        
        self.users: List[UserProfile] = []
        self.items: List[ItemProfile] = []
        self.train_interactions: List[Interaction] = []
        self.test_interactions: List[Interaction] = []
        self.social_graph: Dict[int, List[int]] = {}
        
        self.cf: Optional[CollaborativeFilter] = None
        self.mf: Optional[MatrixFactorization] = None
        self.bert: Optional[BERT4RecTrainer] = None
        self.gnn: Optional[GNNTrainer] = None
        self.mmoe: Optional[MMoETrainer] = None
        self.cold_start: Optional[ColdStartHandler] = None
        
        self.feature_store = FeatureStore()

    def run(self, quick: bool = False) -> dict:
        """
        Executes all pipeline training stages sequentially.
        
        Args:
            quick: If True, caps dataset size to accelerate execution.
        """
        start_time = time.time()
        
        if quick:
            logger.info("Running pipeline in QUICK mode (capping limits)...")
            self.num_users = min(self.num_users, 500)
            self.num_items = min(self.num_items, 2000)
            self.num_interactions = min(self.num_interactions, 20000)
            
        print("=" * 60)
        print("YOUTUBE RECOMMENDATION LITE — TRAINING PIPELINE")
        print("=" * 60)
        
        self._generate()
        self._train_cf()
        self._train_mf()
        self._train_bert()
        self._train_gnn()
        self._train_cold_start()
        self._train_mmoe()
        
        eval_results = self._evaluate()
        self._populate_store()
        
        elapsed = time.time() - start_time
        logger.info(f"Pipeline done in {elapsed:.2f} seconds")
        
        return {
            "elapsed_seconds": elapsed,
            "data_stats": {
                "num_users": len(self.users),
                "num_items": len(self.items),
                "train_interactions": len(self.train_interactions),
                "test_interactions": len(self.test_interactions)
            },
            "evaluation_results": eval_results
        }

    def _generate(self):
        """[Stage 1/7] Synthesizes user preferences, video catalogs, and social nets."""
        logger.info("[1/7] Generating data...")
        generator = DataGenerator(seed=settings.random_seed)
        
        self.users = generator.generate_users(n=self.num_users)
        self.items = generator.generate_items(n=self.num_items)
        
        raw_interactions = generator.generate_interactions(
            users=self.users,
            items=self.items,
            n=self.num_interactions
        )
        
        # Sort interactions by timestamp chronologically
        raw_interactions = sorted(raw_interactions, key=lambda x: getattr(x, 'timestamp', 0.0))
        
        # Split 80/20: first 80% train interactions, last 20% test validation interactions
        split_idx = int(len(raw_interactions) * 0.8)
        self.train_interactions = raw_interactions[:split_idx]
        self.test_interactions = raw_interactions[split_idx:]
        
        # Generate user-user connection graph
        self.social_graph = generator.generate_social_graph(users=self.users)

    def _train_cf(self):
        """[Stage 2/7] Fits user similarity matrices using item-based Collaborative Filtering."""
        logger.info("[2/7] Collaborative Filtering...")
        self.cf = CollaborativeFilter(
            num_users=self.num_users,
            num_items=self.num_items,
            num_neighbors=settings.cf_num_neighbors
        )
        self.cf.fit(self.train_interactions)

    def _train_mf(self):
        """[Stage 3/7] Learns user-item latent factor models using Alternating Least Squares (ALS)."""
        logger.info("[3/7] Matrix Factorization (ALS)...")
        self.mf = MatrixFactorization(
            num_users=self.num_users,
            num_items=self.num_items,
            num_factors=settings.mf_num_factors,
            iterations=settings.mf_epochs
        )
        self.mf.fit(self.train_interactions)

    def _train_bert(self):
        """[Stage 4/7] Trains sequential transformer network for session sequence predictions."""
        logger.info("[4/7] BERT4Rec (Transformer)...")
        self.bert = BERT4RecTrainer(num_items=self.num_items)
        self.bert.train(self.train_interactions, epochs=settings.bert_epochs)

    def _train_gnn(self):
        """[Stage 5/7] Trains message-propagation GraphSAGE network mapping user-friend viewing paths."""
        logger.info("[5/7] Graph Neural Network...")
        self.gnn = GNNTrainer(num_users=self.num_users, num_items=self.num_items)
        self.gnn.build_graph(self.train_interactions, self.social_graph)
        self.gnn.train(self.train_interactions, epochs=settings.gnn_epochs)

    def _train_cold_start(self):
        """[Stage 6/7] Deploys category popularity and similarity models for new users."""
        logger.info("[6/7] Cold Start Models...")
        self.cold_start = ColdStartHandler()
        self.cold_start.fit(self.train_interactions, self.items)

    def _train_mmoe(self):
        """[Stage 7/7] Compiles retrieval embeddings and trains Multi-gate Mixture-of-Experts multitask ranker."""
        logger.info("[7/7] Multi-Objective Ranker (MMoE)...")
        self.mmoe = MMoETrainer(input_dim=FEATURE_DIM)
        
        try:
            X, y = self.mmoe.generate_training_data(
                interactions=self.train_interactions,
                mf_model=self.mf,
                bert_trainer=self.bert,
                gnn_trainer=self.gnn,
                users=self.users,
                items=self.items
            )
            
            if X is not None and len(X) > 0:
                self.mmoe.train(X, y, epochs=settings.mmoe_epochs)
            else:
                logger.warning("Insufficient interaction features to fit MMoE ranker. Skipping fit.")
        except Exception as e:
            logger.error(f"Failed fitting Multi-Objective neural ranker: {e}")

    def _evaluate(self) -> dict:
        """Calculates Precision@10 and NDCG@10 metrics across Collaborative Filtering and ALS models."""
        logger.info("Evaluating models...")
        
        # Build test ground truth mapping: user_id -> set of item_ids watched in test slice
        test_truth = defaultdict(set)
        for inter in self.test_interactions:
            if inter.weight > 1.0:  # Keep clicks, likes, completions as valid engagement
                test_truth[inter.user_id].add(inter.item_id)
                
        # Get category index mapping for catalog representation
        item_cats = {item.item_id: item.category_id for item in self.items}
        
        # Sample evaluation users containing at least 2 target engagements
        eval_users = [
            uid for uid, item_set in test_truth.items()
            if len(item_set) >= 2
        ]
        
        if len(eval_users) > 300:
            rng = np.random.RandomState(settings.random_seed)
            eval_users = list(rng.choice(eval_users, size=300, replace=False))
            
        cf_metrics_list = []
        mf_metrics_list = []
        
        for uid in eval_users:
            rel = test_truth[uid]
            
            # Predict top-20 recommendations from CF
            cf_recs = []
            if self.cf:
                try:
                    cf_recs = [item.item_id for item in self.cf.predict(uid, num_candidates=20)]
                except Exception as e:
                    logger.debug(f"CF evaluation predict failed for user {uid}: {e}")
            cf_metrics = Metrics.all_metrics(cf_recs, rel, k=10, item_cats=item_cats)
            cf_metrics_list.append(cf_metrics)
            
            # Predict top-20 recommendations from ALS MF
            mf_recs = []
            if self.mf:
                try:
                    mf_recs = [item.item_id for item in self.mf.predict(uid, num_candidates=20)]
                except Exception as e:
                    logger.debug(f"MF evaluation predict failed for user {uid}: {e}")
            mf_metrics = Metrics.all_metrics(mf_recs, rel, k=10, item_cats=item_cats)
            mf_metrics_list.append(mf_metrics)
            
        # Average results across all sampled users
        results = {}
        
        if cf_metrics_list:
            cf_avg = {}
            for key in cf_metrics_list[0].keys():
                cf_avg[key] = float(np.mean([m[key] for m in cf_metrics_list]))
            results["CF"] = cf_avg
            logger.info(f"CF — NDCG@10: {cf_avg.get('ndcg@10', 0.0):.4f}, Precision@10: {cf_avg.get('precision@10', 0.0):.4f}")
        else:
            results["CF"] = {}
            
        if mf_metrics_list:
            mf_avg = {}
            for key in mf_metrics_list[0].keys():
                mf_avg[key] = float(np.mean([m[key] for m in mf_metrics_list]))
            results["MF"] = mf_avg
            logger.info(f"MF — NDCG@10: {mf_avg.get('ndcg@10', 0.0):.4f}, Precision@10: {mf_avg.get('precision@10', 0.0):.4f}")
        else:
            results["MF"] = {}
            
        return results

    def _populate_store(self):
        """Indexes user profiles, video metadata, and trained latent embeddings into local FakeRedis."""
        logger.info("Populating feature store...")
        self.feature_store.populate(
            users=self.users,
            items=self.items,
            mf_model=self.mf,
            cf_model=self.cf
        )


# ==========================================
# 🔄 Backward Compatibility Legacy Engine
# ==========================================

class ModelTrainingPipeline:
    """
    Legacy Model Training coordinator triggers fits across central registries.
    """
    def __init__(self):
        self.registry = model_registry

    def execute_training_cycle(self, data_dir: str = "data") -> bool:
        """Orchestrates model fits recording benchmarks."""
        print("🕒 Initializing scheduled Model Training Pipeline...")
        start_time = time.time()
        try:
            self.registry.fit_all(data_dir=data_dir)
            elapsed = time.time() - start_time
            print(f"✅ Training Pipeline run successful! Elapsed: {elapsed:.2f} seconds.")
            return True
        except Exception as e:
            print(f"❌ Training Pipeline execution failed! Details: {str(e)}")
            return False
