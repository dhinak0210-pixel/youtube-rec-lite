import pytest
import numpy as np
import torch
import pandas as pd

from src.models.collaborative_filtering import CollaborativeFilteringRecommender
from src.models.matrix_factorization_als import ALSMatrixFactorization
from src.models.bert4rec import BERT4RecModel
from src.models.gnn_recommender import BipartiteGCNLayer, BipartiteGNNModel
from src.models.mmoe_ranking import MMoEModel

# Mock interactive data fixture
@pytest.fixture
def sample_interactions():
    return pd.DataFrame({
        "user_id": np.random.randint(0, 10, size=100),
        "video_id": np.random.randint(1, 20, size=100),
        "click": np.random.choice([0, 1], size=100),
        "watch_ratio": np.random.uniform(0.0, 1.0, size=100),
        "like": np.random.choice([0, 1], size=100)
    })

def test_collaborative_filtering(sample_interactions):
    recommender = CollaborativeFilteringRecommender(kind="item", k=2)
    recommender.fit(sample_interactions)
    
    # Check if mappings are populated
    assert len(recommender.video_to_idx) > 0
    
    # Check retrieval outputs
    user_id = sample_interactions["user_id"].iloc[0]
    recs = recommender.retrieve_candidates(user_id, top_n=3)
    assert isinstance(recs, list)
    if recs:
        assert isinstance(recs[0], tuple)
        assert len(recs[0]) == 2

def test_als_matrix_factorization(sample_interactions):
    recommender = ALSMatrixFactorization(latent_dims=8, epochs=2)
    recommender.fit(sample_interactions)
    
    assert recommender.user_factors is not None
    assert recommender.item_factors is not None
    assert recommender.user_factors.shape[1] == 8
    
    user_id = sample_interactions["user_id"].iloc[0]
    recs = recommender.retrieve_candidates(user_id, top_n=2)
    assert len(recs) <= 2

def test_bert4rec_model():
    vocab_size = 50
    model = BERT4RecModel(vocab_size=vocab_size, d_model=16, max_seq_len=10)
    
    # Batch of 4, sequence len of 10
    x = torch.randint(0, vocab_size, (4, 10))
    logits = model(x)
    
    assert logits.shape == (4, 10, vocab_size)

def test_custom_gnn():
    num_nodes = 30
    model = BipartiteGNNModel(num_nodes=num_nodes, embedding_dim=12)
    
    # Bipartite edges connecting nodes
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ], dtype=torch.long)
    
    embeddings = model(edge_index)
    assert embeddings.shape == (num_nodes, 12)

def test_mmoe_ranking():
    input_dim = 15
    model = MMoEModel(input_dim=input_dim, num_experts=3, expert_hidden=16, tower_hidden=8)
    
    # Batch of 5 inputs
    x = torch.randn(5, input_dim)
    pred_click, pred_watch = model(x)
    
    assert pred_click.shape == (5,)
    assert pred_watch.shape == (5,)

def test_multi_objective_ranking_model():
    from src.models.mmoe_ranking import MultiObjectiveRankingModel
    input_dim = 20
    model = MultiObjectiveRankingModel(input_dim=input_dim, num_experts=3, expert_hidden=32)
    
    # Batch of 4 inputs
    x = torch.randn(4, input_dim)
    outputs = model(x)
    
    # Assert predictions mapping and shapes
    assert isinstance(outputs, dict)
    for task in ['click', 'watch_complete', 'like', 'dislike']:
        assert task in outputs
        assert outputs[task].shape == (4,)
        
    # Assert blended ranking utility scores
    scores = model.get_final_score(outputs)
    assert scores.shape == (4,)
    
    # Assert multi-task loss computations
    targets = {
        'click': torch.rand(4),
        'watch_complete': torch.rand(4),
        'like': torch.rand(4),
        'dislike': torch.rand(4)
    }
    total_loss, loss_breakdown = model.compute_loss(outputs, targets)
    assert total_loss.item() > 0.0
    assert 'total' in loss_breakdown
    assert 'watch_complete' in loss_breakdown

def test_streaming_pipeline():
    import time
    from src.streaming.simulator import StreamEvent, EventQueue, StreamProcessor
    from src.streaming.redis_client import MockRedisClient
    
    queue = EventQueue(maxlen=1000)
    redis_mock = MockRedisClient()
    processor = StreamProcessor(queue, redis_mock)
    
    # Assert initial conditions
    assert queue.stats()["queue_size"] == 0
    
    # 1. Test production into Kafka queue
    event = StreamEvent(
        user_id=1,
        item_id=10,
        event_type="click",
        timestamp=time.time(),
        session_id="session_1",
        watch_percentage=50.0,
        context={"category": "Gaming"}
    )
    queue.produce(event)
    assert queue.stats()["queue_size"] == 1
    
    # 2. Test consumption batch
    batch = queue.consume_batch(batch_size=10)
    assert len(batch) == 1
    assert queue.stats()["queue_size"] == 0
    assert batch[0].user_id == 1
    
    # 3. Test sliding window updates on StreamProcessor
    processor.item_categories[10] = "Gaming"
    now = time.time()
    processor._update_session(event, now)
    processor._update_item_popularity(event, now)
    processor._update_user_mood(event, now)
    
    assert processor.get_user_current_mood(1) == "Gaming"
    assert len(processor.get_user_session(1)) == 1
    assert processor.get_user_session(1)[0] == 10
    
    # Test trending items
    trending = processor.get_trending_items(top_k=5)
    assert len(trending) == 1
    assert trending[0][0] == 10  # Video 10 is trending

def test_recommender_engine():
    import pandas as pd
    from src.models.recommender_engine import RecommendationEngine
    from src.data_pipeline.preprocessors import RecommenderPreprocessor
    from src.cold_start.handler import ColdStartHandler
    from src.streaming.simulator import StreamProcessor, EventQueue
    from src.streaming.redis_client import MockRedisClient
    from src.models.mmoe_ranking import MultiObjectiveRankingModel
    
    # 1. Mock DataFrames
    users_df = pd.DataFrame({
        "user_id": [1, 2],
        "gender": ["M", "F"],
        "country": ["US", "IN"],
        "age": [25, 30]
    })
    videos_df = pd.DataFrame({
        "video_id": [10, 20],
        "category": ["Gaming", "Tech"],
        "duration": [120, 180]
    })
    
    preprocessor = RecommenderPreprocessor()
    preprocessor.fit(users_df, videos_df)
    
    cold_handler = ColdStartHandler(users_df, videos_df)
    
    queue = EventQueue(maxlen=10)
    redis_mock = MockRedisClient()
    processor = StreamProcessor(queue, redis_mock)
    
    mmoe_model = MultiObjectiveRankingModel(input_dim=7, num_experts=2, expert_hidden=16)
    
    config = {
        "preprocessor": preprocessor,
        "cold_start_handler": cold_handler,
        "stream_processor": processor,
        "mmoe_model": mmoe_model,
        "users_df": users_df,
        "videos_df": videos_df
    }
    
    engine = RecommendationEngine(config)
    
    # Assert candidate generation for mock users
    candidates, sources, counts = engine.generate_candidates(user_id=1, n=10)
    assert isinstance(candidates, list)
    
    # Assert rankings
    ranked = engine.rank_candidates(user_id=1, candidates=[10, 20])
    assert len(ranked) > 0
    assert ranked[0][0] in [10, 20]
    
    # Assert full recommendation response wrapper
    res = engine.recommend(user_id=1, n=2)
    assert res.user_id == 1
    assert len(res.recommendations) > 0
    
    # Assert stats tracking
    stats = engine.performance_stats()
    assert stats["total_queries"] == 1
    assert "p50_latency_ms" in stats

def test_new_cold_start_models():
    from data.schemas import UserProfile, ItemProfile, Interaction, InteractionType
    from models.cold_start import PopularityModel, ContentModel, LinUCBBandit, ColdStartHandler

    # 1. Spawn Mock Profiles
    users = [
        UserProfile(user_id=1, age_bucket=2, gender=1, country_id=10, signup_days_ago=100, num_interactions=10, preferred_categories=[1, 2]),
        UserProfile(user_id=2, age_bucket=4, gender=0, country_id=20, signup_days_ago=200, num_interactions=2, preferred_categories=[3])
    ]
    items = [
        ItemProfile(item_id=100, category_id=1, duration_seconds=120, upload_days_ago=10, creator_id=5, content_embedding=[1.0, 0.0]),
        ItemProfile(item_id=200, category_id=3, duration_seconds=180, upload_days_ago=20, creator_id=6, content_embedding=[0.0, 1.0])
    ]
    interactions = [
        Interaction(user_id=1, item_id=100, interaction_type=InteractionType.CLICK),
        Interaction(user_id=2, item_id=200, interaction_type=InteractionType.LIKE)
    ]

    # 2. Test Popularity Model
    pop = PopularityModel()
    pop.fit(interactions, items)
    assert pop._fitted is True
    recs = pop.recommend(user=users[0], n=2)
    assert len(recs) == 2
    assert recs[0].source == "popularity"

    # 3. Test Content Model
    content = ContentModel()
    content.fit(items)
    sims = content.similar(seeds=[100], n=1)
    assert len(sims) == 1
    assert sims[0].source == "content"

    # 4. Test LinUCB Bandit
    bandit = LinUCBBandit(alpha=1.0)
    bandit_recs = bandit.select(user=users[0], candidates=[100, 200], n=2)
    assert len(bandit_recs) == 2
    assert bandit_recs[0].source == "bandit"

    # 5. Test Context Updates
    bandit.update(item_id=100, user=users[0], reward=1.0)
    assert np.any(bandit.b[100] > 0)
    assert bandit.b[100].shape == (8,)

    # 6. Test Handler Orchestrator
    handler = ColdStartHandler()
    handler.fit(interactions, items)
    final_recs = handler.recommend(user=users[1], n=2)
    assert len(final_recs) == 2
    assert final_recs[0].source == "bandit"

def test_model_optimization():
    import torch
    import torch.nn as nn
    from models.optimization import DynamicQuantizer, LRUEmbeddingCache, PerformanceProfiler

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = DummyModel()
    
    # Test Size MB
    size_mb = DynamicQuantizer.get_model_size_mb(model)
    assert size_mb > 0.0

    # Test Quantization
    q_model = DynamicQuantizer.quantize(model)
    assert q_model is not None

    # Test Cache
    cache = LRUEmbeddingCache(capacity=2, ttl_seconds=10)
    cache.put("user_1", [0.1, 0.2])
    assert cache.get("user_1") == [0.1, 0.2]
    
    # Test Capacity Eviction
    cache.put("user_2", [0.3])
    cache.put("user_3", [0.4])
    assert cache.get("user_1") is None  # Evicted because capacity is 2!
    assert cache.get("user_3") == [0.4]

    # Test Benchmarking
    x = torch.randn(5, 10)
    bench = PerformanceProfiler.benchmark_inference(model, x, num_runs=5)
    assert "p50_latency_ms" in bench
    assert bench["qps_throughput"] > 0





