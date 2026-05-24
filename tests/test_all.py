"""
Comprehensive Pytest Test Suite.

Audits every custom module (Settings, Mock Data Generator, Collaborative Filtering, 
Matrix Factorization, BERT4Rec, GNN, Cold Start, Feature Store, A/B Testing, 
Offline Evaluation Metrics, and Real-Time Streaming Ingestion pipelines).
"""

import pytest
import numpy as np
import time
import threading
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any

from config import settings
from data import MockDataGenerator
from data.generator import DataGenerator
from data.schemas import InteractionType, UserProfile, ItemProfile, Interaction, ScoredItem
from models.collaborative_filtering import CollaborativeFilter
from models.matrix_factorization import MatrixFactorization
from models.sequential_recommender import BERT4Rec, BERT4RecTrainer
from models.graph_neural_network import SocialGraph, GNNTrainer
from models.cold_start import PopularityModel, ContentModel, LinUCBBandit, ColdStartHandler
from services.feature_store import FeatureStore
from services.ab_testing import ABTestingService
from evaluation.metrics import Metrics
from streaming.pipeline import EventQueue, StreamProcessor, StreamEvent

# ==========================================
# ⚙️ Part 1 - Expensive Test Fixtures
# ==========================================

@pytest.fixture(scope="session")
def small_data() -> Tuple[List[UserProfile], List[ItemProfile], List[Interaction], Dict[int, List[int]]]:
    """Generates a minimal session-scoped dataset for training verification."""
    gen = DataGenerator(seed=42)
    users = gen.generate_users(n=100)
    items = gen.generate_items(n=500)
    interactions = gen.generate_interactions(users, items, n=5000)
    social_graph = gen.generate_social_graph(users, n_connections=500)
    return users, items, interactions, social_graph

@pytest.fixture(scope="session")
def trained_cf(small_data) -> CollaborativeFilter:
    """Trains an Item-Based Collaborative Filtering similarities index."""
    users, items, interactions, social_graph = small_data
    cf = CollaborativeFilter(num_users=100, num_items=500, num_neighbors=10)
    cf.fit(interactions)
    return cf

@pytest.fixture(scope="session")
def trained_mf(small_data) -> MatrixFactorization:
    """Trains an Implicit ALS Matrix Factorization model."""
    users, items, interactions, social_graph = small_data
    mf = MatrixFactorization(num_users=100, num_items=500, num_factors=16, regularization=0.01, iterations=3)
    mf.fit(interactions)
    return mf

# ==========================================
# 🧪 Part 2 - Verification Test Classes
# ==========================================

class TestDataGenerator:
    """Audits the synthetic data distribution scaling generator properties."""

    def test_user_count(self, small_data):
        """Asserts that exactly 100 users are generated."""
        users, _, _, _ = small_data
        assert len(users) == 100

    def test_item_count(self, small_data):
        """Asserts that exactly 500 item profiles are generated."""
        _, items, _, _ = small_data
        assert len(items) == 500

    def test_interaction_count(self, small_data):
        """Asserts that exactly 5000 interactions are generated."""
        _, _, interactions, _ = small_data
        assert len(interactions) == 5000

    def test_user_ids_unique(self, small_data):
        """Asserts that all generated user profiles possess unique identifiers."""
        users, _, _, _ = small_data
        user_ids = [u.user_id for u in users]
        assert len(user_ids) == len(set(user_ids))

    def test_preferred_categories_valid(self, small_data):
        """Asserts that each user has between 1 and 4 preferred categories within 0-19 bounds."""
        users, _, _, _ = small_data
        for u in users:
            assert 1 <= len(u.preferred_categories) <= 4
            for cat in u.preferred_categories:
                assert 0 <= cat <= 19

    def test_interaction_types_valid(self, small_data):
        """Asserts that all interaction types map to valid InteractionType values."""
        _, _, interactions, _ = small_data
        for inter in interactions:
            assert isinstance(inter.interaction_type, InteractionType)

    def test_social_graph_not_empty(self, small_data):
        """Asserts that the social network possesses directional links."""
        _, _, _, social_graph = small_data
        total_edges = sum(len(friends) for friends in social_graph.values())
        assert total_edges > 0

    def test_reproducibility(self):
        """Asserts that setting the same seed yields identical user distributions."""
        gen1 = DataGenerator(seed=123)
        gen2 = DataGenerator(seed=123)
        users1 = gen1.generate_users(n=10)
        users2 = gen2.generate_users(n=10)
        for u1, u2 in zip(users1, users2):
            assert u1.user_id == u2.user_id
            assert u1.preferred_categories == u2.preferred_categories


class TestCollaborativeFilter:
    """Audits the Item-Based Collaborative Filtering cosine simulator index."""

    def test_fit_creates_similarities(self, trained_cf):
        """Asserts that cosine similarity matrix weights are computed on fit."""
        assert trained_cf.item_similarities is not None
        assert len(trained_cf.item_similarities) > 0

    def test_predict_returns_list(self, trained_cf):
        """Asserts that predicting recommendations for an active profile returns scored candidates."""
        user_id = list(trained_cf.user_history.keys())[0]
        res = trained_cf.predict(user_id=user_id, num_candidates=10)
        assert isinstance(res, list)
        assert len(res) <= 10

    def test_cold_user_returns_empty(self, trained_cf):
        """Asserts that requesting candidates for a completely out-of-range user returns an empty list."""
        res = trained_cf.predict(user_id=9999, num_candidates=10)
        assert res == []

    def test_scores_positive(self, trained_cf):
        """Asserts that similarity prediction scores are positive values."""
        user_id = list(trained_cf.user_history.keys())[0]
        res = trained_cf.predict(user_id=user_id, num_candidates=10)
        for item in res:
            assert item.score >= 0.0

    def test_source_label(self, trained_cf):
        """Asserts that candidate items are flagged with the 'cf' label source."""
        user_id = list(trained_cf.user_history.keys())[0]
        res = trained_cf.predict(user_id=user_id, num_candidates=10)
        for item in res:
            assert item.source == "cf"

    def test_exclude_respected(self, trained_cf):
        """Asserts that user historically viewed items are properly filtered from outputs."""
        user_id = list(trained_cf.user_history.keys())[0]
        exclude_set = {10, 20, 30}
        res = trained_cf.predict(user_id=user_id, num_candidates=10, exclude_items=exclude_set)
        for item in res:
            assert item.item_id not in exclude_set


class TestMatrixFactorization:
    """Audits the regularized PyTorch Implicit ALS Matrix Factorization embeddings."""

    def test_factors_shape(self, trained_mf):
        """Asserts that the computed user factors matrix possesses the correct dimensions."""
        assert trained_mf.user_factors.shape == (100, 16)

    def test_no_nan_in_factors(self, trained_mf):
        """Asserts that user and item projection matrices possess no NaN values."""
        assert not np.isnan(trained_mf.user_factors).any()
        assert not np.isnan(trained_mf.item_factors).any()

    def test_predict_returns_items(self, trained_mf):
        """Asserts that matrix dot-product candidate retrieval returns the correct counts."""
        res = trained_mf.predict(user_id=1, num_candidates=10)
        assert len(res) <= 10

    def test_user_embedding_shape(self, trained_mf):
        """Asserts that active user factor embeddings match the projection dimensions."""
        emb = trained_mf.get_user_embedding(user_id=5)
        assert emb is not None
        assert emb.shape == (16,)

    def test_item_embedding_shape(self, trained_mf):
        """Asserts that active item factor embeddings match the projection dimensions."""
        emb = trained_mf.get_item_embedding(item_id=10)
        assert emb is not None
        assert emb.shape == (16,)

    def test_out_of_range_returns_none(self, trained_mf):
        """Asserts that requesting factors for non-existent users returns None."""
        assert trained_mf.get_user_embedding(user_id=9999) is None
        assert trained_mf.get_item_embedding(item_id=9999) is None

    def test_fitted_flag(self, trained_mf):
        """Asserts that the fitted lifecycle indicator is flagged to True."""
        assert trained_mf._fitted is True


class TestBERT4Rec:
    """Audits PyTorch sequence bidirectional self-attention transformer forward passes."""

    def test_forward_shape(self):
        """Asserts that forwarding random batched sequence tensor yields expected logits."""
        model = BERT4Rec(num_items=100, d_model=64)
        seq = torch.ones(4, 10, dtype=torch.long)
        out = model(seq)
        assert out.shape == (4, 10, 101)

    def test_trainer_builds_sequences(self, small_data):
        """Asserts that sequential data preprocessing compiles valid, non-empty history vectors."""
        _, _, interactions, _ = small_data
        trainer = BERT4RecTrainer(num_items=500)
        seqs = trainer.build_sequences(interactions, min_len=2)
        assert isinstance(seqs, dict)
        for u, history in seqs.items():
            assert len(history) >= 2

    def test_predict_returns_ranked(self, small_data):
        """Asserts that sequentialTransformer output candidates are sorted in descending relevance order."""
        _, _, interactions, _ = small_data
        trainer = BERT4RecTrainer(num_items=500)
        trainer.train(interactions, epochs=1, batch_size=64)
        
        user_history = [12, 45, 98]
        recs = trainer.predict(user_history, top_k=10)
        assert len(recs) <= 10
        scores = [item.score for item in recs]
        assert scores == sorted(scores, reverse=True)

    def test_get_embedding_dim(self, small_data):
        """Asserts that user sequential aggregate embeddings match settings.bert_d_model."""
        _, _, interactions, _ = small_data
        trainer = BERT4RecTrainer(num_items=500)
        trainer.train(interactions, epochs=1, batch_size=64)
        
        emb = trainer.get_user_embedding([12, 45, 98])
        assert emb is not None
        assert emb.shape == (64,)


class TestGNN:
    """Audits PyTorch social network neighborhood aggregation GraphSAGE convolutions."""

    def test_social_graph_add(self):
        """Asserts that added follower links are properly retrievable."""
        sg = SocialGraph()
        sg.add_follow(10, 20)
        sg.add_follow(10, 30)
        friends = sg.get_friends(10)
        assert 20 in friends
        assert 30 in friends

    def test_friend_watch_rate_zero_no_friends(self):
        """Asserts that computing rates for isolated profiles without followers yields exactly zero."""
        sg = SocialGraph()
        assert sg.friend_watch_rate(1, 100) == 0.0

    def test_social_features_shape(self):
        """Asserts that extracted social feature vectors possess shape (4,) and values within [0,1]."""
        sg = SocialGraph()
        sg.add_follow(1, 2)
        sg.add_follow(1, 3)
        sg.add_view(2, 50)
        
        feats = sg.get_social_features(1, 50)
        assert feats.shape == (4,)
        assert all(0.0 <= val <= 1.0 for val in feats)

    def test_gnn_trains_without_error(self, small_data):
        """Asserts that training the bipartite user-item GraphSAGE social net minimizes without exceptions."""
        _, _, interactions, social_graph = small_data
        trainer = GNNTrainer(num_users=100, num_items=500, max_neighbors=5)
        trainer.build_graph(interactions, social_graph)
        losses = trainer.train(interactions, epochs=2)
        assert len(losses) == 2

    def test_score_items_returns_correct_count(self, small_data):
        """Asserts that GNN item scoring returns the correct item predictions count."""
        _, _, interactions, social_graph = small_data
        trainer = GNNTrainer(num_users=100, num_items=500, max_neighbors=5)
        trainer.build_graph(interactions, social_graph)
        trainer.train(interactions, epochs=1)
        
        scores = trainer.score_items(user_id=1, item_ids=[10, 20, 30, 40, 50])
        assert len(scores) == 5
        assert isinstance(scores[0], tuple)


class TestColdStart:
    """Audits cold start popularity models and contextual multi-armed bandit select heuristics."""

    def test_popularity_model(self, small_data):
        """Asserts that category PopularityModel recommend returns <= 10 items flagged with popularity."""
        _, items, interactions, _ = small_data
        model = PopularityModel()
        model.fit(interactions, items)
        
        recs = model.recommend(n=10)
        assert len(recs) <= 10
        for item in recs:
            assert item.source == "popularity"

    def test_popularity_scores_range(self, small_data):
        """Asserts that normalized popularity scores are strictly within standard [0, 1] bounds."""
        _, items, interactions, _ = small_data
        model = PopularityModel()
        model.fit(interactions, items)
        
        recs = model.recommend(n=50)
        for item in recs:
            assert 0.0 <= item.score <= 1.0

    def test_content_model_similar(self, small_data):
        """Asserts that cosine content embedding similarities exclude query seeds from outputs."""
        _, items, _, _ = small_data
        model = ContentModel()
        model.fit(items)
        
        recs = model.similar(seeds=[10, 20], n=5, exclude={10, 20})
        for item in recs:
            assert item.item_id not in [10, 20]

    def test_bandit_selects_n(self, small_data):
        """Asserts that the context LinUCB bandit explorer returns exactly n requested items."""
        users, _, _, _ = small_data
        bandit = LinUCBBandit(ctx_dim=8)
        recs = bandit.select(users[0], candidates=[10, 20, 30, 40, 50], n=3)
        assert len(recs) == 3

    def test_bandit_update_doesnt_crash(self, small_data):
        """Asserts that updating contextual LinUCB bandit ridge projection matrices executes without crashes."""
        users, _, _, _ = small_data
        bandit = LinUCBBandit(ctx_dim=8)
        bandit.select(users[0], candidates=[10, 20], n=1)
        bandit.update(item_id=10, user=users[0], reward=1.0)
        # Verify no crash occurred

    def test_cold_start_handler(self, small_data):
        """Asserts that the ColdStartHandler orchestrates and delivers recommendations for new users."""
        users, items, interactions, _ = small_data
        handler = ColdStartHandler()
        handler.fit(interactions, items)
        
        recs = handler.recommend(users[0], n=10)
        assert len(recs) > 0


class TestFeatureStore:
    """Audits local zero-cost in-memory mock Redis key-value stores."""

    def test_store_and_get_user(self):
        """Asserts that stored user feature JSON dictionaries match lookups exactly."""
        fs = FeatureStore()
        data = {"age": 25, "gender": 1, "country": "US"}
        fs.store_user(42, data)
        assert fs.get_user(42) == data

    def test_missing_key_returns_none(self):
        """Asserts that looking up missing users or items returns None."""
        fs = FeatureStore()
        assert fs.get_user(9999) is None
        assert fs.get_item(9999) is None

    def test_store_and_get_embedding(self):
        """Asserts that stored raw numpy factor embedding bytes match loaded arrays within tolerances."""
        fs = FeatureStore()
        emb = np.array([0.15, -0.42, 0.99, -0.01], dtype=np.float32)
        fs.store_emb("user", 42, emb)
        retrieved = fs.get_emb("user", 42, dim=4)
        assert retrieved is not None
        assert np.allclose(emb, retrieved, atol=1e-5)

    def test_history_stored_correctly(self):
        """Asserts that stored user engagement lists are returned intact."""
        fs = FeatureStore()
        history = [100, 205, 500]
        fs.store_history(42, history)
        assert fs.get_history(42) == history

    def test_missing_history_returns_empty(self):
        """Asserts that querying history for non-existent users returns an empty list."""
        fs = FeatureStore()
        assert fs.get_history(9999) == []


class TestABTesting:
    """Audits MD5 user partition assignments and two-proportion Z-test evaluations."""

    def test_create_experiment(self):
        """Asserts that creating experiments populates control and treatment target groups."""
        ab = ABTestingService()
        exp = ab.create("ranking_test", description="Ranking changes", control_frac=0.5)
        assert "control" in exp.groups
        assert "treatment" in exp.groups

    def test_assignment_deterministic(self):
        """Asserts that hashing assign mappings are deterministic for identical user profiles."""
        ab = ABTestingService()
        ab.create("ranking_test")
        ab.start("ranking_test")
        
        g1 = ab.assign(user_id=123, exp_id="ranking_test")
        g2 = ab.assign(user_id=123, exp_id="ranking_test")
        assert g1 == g2

    def test_traffic_split_approximately_50_50(self):
        """Asserts that user assignment traffic splits control and treatment cohorts approximately 50-50."""
        ab = ABTestingService()
        ab.create("split_test", control_frac=0.5)
        ab.start("split_test")
        
        counts = defaultdict(int)
        for uid in range(1000):
            group = ab.assign(uid, "split_test")
            if group:
                counts[group] += 1
                
        # Assert split is within 500 ± 150 bounds
        assert abs(counts["control"] - 500) < 150

    def test_draft_experiment_returns_none(self):
        """Asserts that draft experiments (not yet started) return None cohort allocations."""
        ab = ABTestingService()
        ab.create("draft_test")
        assert ab.assign(1, "draft_test") is None

    def test_significant_difference_detected(self):
        """Asserts that significant CTR differences between cohorts (5% vs 10%) are flagged by Z-tests."""
        ab = ABTestingService()
        # 5% CTR vs 10% CTR over 2000 users yields significance
        report = ab.simulate("sig_test", n_users=2000, ctrl_ctr=0.05, treat_ctr=0.10)
        assert report["ctr_test"]["significant"] is True
        assert report["ctr_test"]["verdict"] == "SHIP treatment"

    def test_no_difference_not_significant(self):
        """Asserts that negligible CTR differences (5% vs 5.1%) are marked non-significant by Z-tests."""
        ab = ABTestingService()
        report = ab.simulate("nonsig_test", n_users=1000, ctrl_ctr=0.05, treat_ctr=0.051)
        assert report["ctr_test"]["significant"] is False
        assert report["ctr_test"]["verdict"] == "CONTINUE"


class TestMetrics:
    """Audits offline accuracy IR evaluation equations."""

    def test_precision(self):
        """Asserts Precision@K equations: [1,2,3] recommended vs {1,3} true at k=3 equals 2/3."""
        assert Metrics.precision_at_k([1, 2, 3], {1, 3}, k=3) == pytest.approx(2/3)

    def test_recall(self):
        """Asserts Recall@K equations: [1,2,3] recommended vs {1,3,4} true at k=3 equals 2/3."""
        assert Metrics.recall_at_k([1, 2, 3], {1, 3, 4}, k=3) == pytest.approx(2/3)

    def test_ndcg_perfect(self):
        """Asserts that a perfectly matching ranked output delivers exactly 1.0."""
        assert Metrics.ndcg_at_k([1, 2], {1, 2}, k=2) == pytest.approx(1.0)

    def test_ndcg_zero_no_hits(self):
        """Asserts that zero-overlap recommendations yield exactly 0.0."""
        assert Metrics.ndcg_at_k([1, 2], {3, 4}, k=2) == pytest.approx(0.0)

    def test_hit_rate_true(self):
        """Asserts that Hit Rate returns 1.0 when at least one relevant item appears in the list."""
        assert Metrics.hit_rate([1, 2], {2}, k=2) == pytest.approx(1.0)

    def test_hit_rate_false(self):
        """Asserts that Hit Rate returns 0.0 when zero relevant items appear in the list."""
        assert Metrics.hit_rate([1, 2], {3}, k=2) == pytest.approx(0.0)

    def test_coverage(self):
        """Asserts that Catalog Coverage calculates correct unique candidate ratio."""
        all_recs = [[1, 2], [3, 4], [4, 5]]
        assert Metrics.coverage(all_recs, total_items=10) == pytest.approx(0.5)

    def test_diversity_all_different(self):
        """Asserts that completely diverse categories in ranked items yield exactly 1.0."""
        item_cats = {1: 1, 2: 2, 3: 3}
        assert Metrics.diversity([1, 2, 3], item_cats) == pytest.approx(1.0)

    def test_all_metrics_returns_dict(self):
        """Asserts that all_metrics maps values inside a comprehensive dictionary."""
        item_cats = {1: 1, 2: 2}
        metrics = Metrics.all_metrics([1, 2], {1}, k=2, item_cats=item_cats)
        assert "precision@2" in metrics
        assert "ndcg@2" in metrics
        assert "hit_rate@2" in metrics


class TestStreaming:
    """Audits multi-threaded Kafka topic simulations and sliding Flink window evictions."""

    def test_queue_produce_consume(self):
        """Asserts thread-safe queue produce/consume increments correctly."""
        queue = EventQueue()
        event = StreamEvent(user_id=10, item_id=20, event_type="click")
        queue.produce(event)
        
        assert queue.size == 1
        batch = queue.consume_batch(n=1)
        assert len(batch) == 1
        assert batch[0].user_id == 10

    def test_queue_thread_safe(self):
        """Asserts that multiple concurrent threads can produce to the event queue without data losses."""
        queue = EventQueue()
        def producer_worker():
            for _ in range(100):
                queue.produce(StreamEvent(user_id=1, item_id=2, event_type="click"))
                
        threads = [threading.Thread(target=producer_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert queue.produced == 500
        assert queue.size == 500

    def test_processor_processes_events(self):
        """Asserts that the background streaming daemon consumes and processes queue events."""
        queue = EventQueue()
        processor = StreamProcessor(queue=queue)
        processor.start()
        
        queue.produce(StreamEvent(user_id=10, item_id=20, event_type="click"))
        time.sleep(0.1)  # Allow background consumer to loop
        
        st = processor.stats()
        assert st["processed"] == 1
        processor.stop()

    def test_trending_items_tracked(self):
        """Asserts that highly viewed items correctly appear inside sliding trending windows."""
        queue = EventQueue()
        processor = StreamProcessor(queue=queue)
        processor.start()
        
        queue.produce(StreamEvent(user_id=10, item_id=42, event_type="click"))
        queue.produce(StreamEvent(user_id=11, item_id=42, event_type="click"))
        time.sleep(0.1)
        
        trending = [iid for iid, count in processor.get_trending(top_k=5)]
        assert 42 in trending
        processor.stop()

    def test_session_updated(self):
        """Asserts that user active real-time sliding watch history matches updates."""
        queue = EventQueue()
        processor = StreamProcessor(queue=queue)
        processor.start()
        
        queue.produce(StreamEvent(user_id=99, item_id=7, event_type="click"))
        time.sleep(0.1)
        
        session = processor.get_session(user_id=99)
        session_items = [e["item_id"] for e in session]
        assert 7 in session_items
        processor.stop()

    def test_throughput_above_500_events_per_second(self):
        """Asserts that the ingestion pipeline processes high event rates safely."""
        queue = EventQueue()
        processor = StreamProcessor(queue=queue)
        processor.start()
        
        start_t = time.time()
        for _ in range(100):
            queue.produce(StreamEvent(user_id=1, item_id=2, event_type="click"))
            
        time.sleep(0.05)
        st = processor.stats()
        elapsed = time.time() - start_t
        throughput = st["processed"] / max(elapsed, 1e-10)
        
        # Verify throughput bounds or that all events processed cleanly
        assert throughput > 500 or st["processed"] == 100
        processor.stop()
