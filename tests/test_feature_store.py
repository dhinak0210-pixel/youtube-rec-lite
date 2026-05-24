import numpy as np
from services.feature_store import FeatureStore
from data.schemas import UserProfile, ItemProfile

class DummyMFModel:
    def __init__(self, num_users=10, num_items=10, factors=8):
        self.user_factors = np.arange(num_users * factors, dtype=np.float32).reshape(num_users, factors)
        self.item_factors = np.arange(num_items * factors, dtype=np.float32).reshape(num_items, factors)
        self.user_history = {i: list(range(i + 1)) for i in range(num_users)}

class DummyCFModel:
    def __init__(self, num_users=10):
        self.user_history = {i: list(range(10 - i)) for i in range(num_users)}

def test_feature_store_basic_operations():
    """Verifies basic set, get, embedding serializations, and TTL cache mappings."""
    fs = FeatureStore()
    
    # 1. Test User Feature Storage
    user_data = {
        "age_bucket": 3,
        "gender": 1,
        "country_id": 45,
        "signup_days_ago": 120,
        "num_interactions": 42,
        "preferred_categories": [1, 5, 8]
    }
    fs.store_user(101, user_data)
    retrieved_user = fs.get_user(101)
    assert retrieved_user == user_data, "User data mismatch!"
    
    # Non-existent user should yield None
    assert fs.get_user(9999) is None

    # 2. Test Item Feature Storage
    item_data = {
        "category_id": 4,
        "duration_seconds": 180,
        "upload_days_ago": 3,
        "like_ratio": 0.95,
        "avg_watch_pct": 0.72,
        "total_views": 1500
    }
    fs.store_item(505, item_data)
    retrieved_item = fs.get_item(505)
    assert retrieved_item == item_data, "Item data mismatch!"
    
    # Non-existent item should yield None
    assert fs.get_item(9999) is None

def test_feature_store_embeddings():
    """Verifies binary representation caching and precise reconstructed numpy vector values."""
    fs = FeatureStore()
    
    # Create random float32 embedding vector
    emb = np.array([0.1, -2.5, 3.14, 0.007, -0.99], dtype=np.float32)
    fs.store_emb("user", 42, emb)
    
    retrieved_emb = fs.get_emb("user", 42, dim=5)
    assert retrieved_emb is not None
    assert np.allclose(retrieved_emb, emb), "Embedding vector reconstructed with floating point deviations!"
    
    # Verify non-existent embeddings return None
    assert fs.get_emb("item", 999, dim=5) is None

def test_feature_store_history_capping():
    """Verifies that lists longer than 100 entries are correctly capped to protect memory footprint."""
    fs = FeatureStore()
    
    # Store list of 150 items
    long_history = list(range(150))
    fs.store_history(202, long_history)
    
    retrieved_history = fs.get_history(202)
    assert len(retrieved_history) == 100, f"History list not capped correctly! Length was {len(retrieved_history)}"
    assert retrieved_history == list(range(100)), "History entries modified during capping!"
    
    # Verify empty list returned for non-existent users
    assert fs.get_history(9999) == []

def test_feature_store_populate():
    """Verifies bulk populate functionality integrating Pydantic schemas and dummy Latent models."""
    fs = FeatureStore()
    
    users = [
        UserProfile(user_id=1, age_bucket=2, gender=1, country_id=10, signup_days_ago=50, num_interactions=12, preferred_categories=[1, 2]),
        UserProfile(user_id=2, age_bucket=4, gender=0, country_id=20, signup_days_ago=150, num_interactions=5, preferred_categories=[0])
    ]
    
    items = [
        ItemProfile(item_id=1, category_id=1, duration_seconds=120, upload_days_ago=2, creator_id=0, total_views=100, like_ratio=0.8, avg_watch_pct=0.6, content_embedding=[0.1, 0.2]),
        ItemProfile(item_id=2, category_id=0, duration_seconds=300, upload_days_ago=10, creator_id=1, total_views=50, like_ratio=0.7, avg_watch_pct=0.4, content_embedding=[-0.1, 0.5])
    ]
    
    mf = DummyMFModel()
    cf = DummyCFModel()
    
    fs.populate(users, items, mf_model=mf, cf_model=cf)
    
    # Verify user embeddings
    user1_emb = fs.get_emb("user", 1, dim=8)
    assert user1_emb is not None
    assert np.allclose(user1_emb, mf.user_factors[1])
    
    # Verify item embeddings
    item2_emb = fs.get_emb("item", 2, dim=8)
    assert item2_emb is not None
    assert np.allclose(item2_emb, mf.item_factors[2])
    
    # Verify user features
    u1_feat = fs.get_user(1)
    assert u1_feat is not None
    assert u1_feat["age_bucket"] == 2
    assert u1_feat["gender"] == 1
    
    # Verify item features
    i2_feat = fs.get_item(2)
    assert i2_feat is not None
    assert i2_feat["duration_seconds"] == 300
    assert i2_feat["total_views"] == 50
    
    # Verify collaborative filter history priority lookup
    u1_hist = fs.get_history(1)
    assert u1_hist == cf.user_history[1], "History should load from cf_model first!"
