"""
High-Performance In-Memory Local Feature Store using FakeRedis.

Provides sub-millisecond storage and retrieval of:
1. User categorical demographics
2. Video catalog metadata
3. Latent embedding arrays (Numpy float32 buffers)
4. Active watch interaction history lists (capped at last 100 items)
Uses an automatic expiration Time-To-Live (TTL) of 3600 seconds.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
import fakeredis
from loguru import logger
from config import settings

class FeatureStore:
    """
    Local in-memory zero-cost Feature Store encapsulating a FakeRedis engine.
    
    Keys Schema Map:
    - User features: `u:feat:{user_id}` -> JSON dict of demographics
    - Item features: `i:feat:{item_id}` -> JSON dict of item attributes
    - Entity embeddings: `{kind}:emb:{entity_id}` -> raw float32 numpy bytes
    - User history list: `u:hist:{user_id}` -> JSON list of recently viewed item IDs
    """

    def __init__(self):
        # Decode responses should be False so we can store raw float32 binary embeddings directly
        self._r = fakeredis.FakeRedis(decode_responses=False)
        self.is_loaded = False
        logger.info("Feature store: using FakeRedis (local, zero cost)")

    def _set(self, key: str, value_bytes: bytes, ttl: int = 3600):
        """Internal helper to set key-value with TTL expiration constraint."""
        self._r.setex(key, ttl, value_bytes)

    def _get(self, key: str) -> Optional[bytes]:
        """Internal helper to get raw byte value from FakeRedis store."""
        return self._r.get(key)

    def store_user(self, user_id: int, data: dict):
        """Serializes user demographic features to JSON bytes and caches under key 'u:feat:{user_id}'."""
        key = f"u:feat:{user_id}"
        val_bytes = json.dumps(data).encode('utf-8')
        self._set(key, val_bytes)

    def get_user(self, user_id: int) -> Optional[dict]:
        """Retrieves and parses cached user demographic features from key 'u:feat:{user_id}'."""
        key = f"u:feat:{user_id}"
        raw = self._get(key)
        if raw is None:
            return None
        return json.loads(raw.decode('utf-8'))

    def store_item(self, item_id: int, data: dict):
        """Serializes video catalog metadata to JSON bytes and caches under key 'i:feat:{item_id}'."""
        key = f"i:feat:{item_id}"
        val_bytes = json.dumps(data).encode('utf-8')
        self._set(key, val_bytes)

    def get_item(self, item_id: int) -> Optional[dict]:
        """Retrieves and parses cached video catalog metadata from key 'i:feat:{item_id}'."""
        key = f"i:feat:{item_id}"
        raw = self._get(key)
        if raw is None:
            return None
        return json.loads(raw.decode('utf-8'))

    def store_emb(self, kind: str, entity_id: int, emb: np.ndarray):
        """Caches raw float32 embedding bytes under key '{kind}:emb:{entity_id}'."""
        key = f"{kind}:emb:{entity_id}"
        value_bytes = emb.astype(np.float32).tobytes()
        self._set(key, value_bytes)

    def get_emb(self, kind: str, entity_id: int, dim: int) -> Optional[np.ndarray]:
        """Retrieves and reconstructs float32 embedding vector from key '{kind}:emb:{entity_id}'."""
        key = f"{kind}:emb:{entity_id}"
        raw = self._get(key)
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    def store_history(self, user_id: int, items: List[int]):
        """Caches last 100 historical watch items as a JSON list under key 'u:hist:{user_id}'."""
        key = f"u:hist:{user_id}"
        recent_items = list(items)[:100]
        val_bytes = json.dumps(recent_items).encode('utf-8')
        self._set(key, val_bytes)

    def get_history(self, user_id: int) -> List[int]:
        """Retrieves user watch interaction history from key 'u:hist:{user_id}', defaulting to empty list."""
        key = f"u:hist:{user_id}"
        raw = self._get(key)
        if raw is None:
            return []
        return json.loads(raw.decode('utf-8'))

    def populate(self, users: List[Any], items: List[Any], mf_model: Optional[Any] = None, cf_model: Optional[Any] = None):
        """
        Bulk populates user/item feature records, latent embeddings, and histories into FakeRedis.
        """
        logger.info("Bulk populating local zero-cost Feature Store...")

        def get_val(obj: Any, key_name: str, fallback: Any = None) -> Any:
            """Safely accesses attribute or dict key on objects."""
            if hasattr(obj, key_name):
                return getattr(obj, key_name)
            if isinstance(obj, dict):
                return obj.get(key_name, fallback)
            return fallback

        # 1. Populate user states
        for u in users:
            uid = get_val(u, "user_id")
            if uid is None:
                continue

            # Cache MF user embeddings if available
            if mf_model and hasattr(mf_model, "user_factors") and mf_model.user_factors is not None:
                if 0 <= uid < mf_model.user_factors.shape[0]:
                    self.store_emb("user", uid, mf_model.user_factors[uid])

            # Cache user demographics
            user_dict = {
                "age_bucket": get_val(u, "age_bucket", 3),
                "gender": get_val(u, "gender", 0),
                "country_id": get_val(u, "country_id", 0),
                "signup_days_ago": get_val(u, "signup_days_ago", 30),
                "num_interactions": get_val(u, "num_interactions", 0),
                "preferred_categories": get_val(u, "preferred_categories", [])
            }
            self.store_user(uid, user_dict)

            # Retrieve and cache watch history list
            hist = []
            if cf_model and hasattr(cf_model, "user_history"):
                hist = list(cf_model.user_history.get(uid, []))
            elif mf_model and hasattr(mf_model, "user_history"):
                hist = list(mf_model.user_history.get(uid, []))
            
            self.store_history(uid, hist)

        # 2. Populate item states
        for i in items:
            iid = get_val(i, "item_id")
            if iid is None:
                continue

            # Cache MF item embeddings if available
            if mf_model and hasattr(mf_model, "item_factors") and mf_model.item_factors is not None:
                if 0 <= iid < mf_model.item_factors.shape[0]:
                    self.store_emb("item", iid, mf_model.item_factors[iid])

            # Cache item characteristics
            item_dict = {
                "category_id": get_val(i, "category_id", 0),
                "duration_seconds": get_val(i, "duration_seconds", 300),
                "upload_days_ago": get_val(i, "upload_days_ago", 10),
                "like_ratio": get_val(i, "like_ratio", 0.5),
                "avg_watch_pct": get_val(i, "avg_watch_pct", 0.5),
                "total_views": get_val(i, "total_views", 0)
            }
            self.store_item(iid, item_dict)

        logger.info("Feature store populated")

    # ==========================================
    # 🔄 Legacy / UI Compatibility Layer Methods
    # ==========================================

    def load_features(self):
        """
        Legacy/UI loader method. Parses data from CSVs and stores them in FakeRedis 
        to ensure all dashboard components can bootstrap without missing features.
        """
        logger.info("FeatureStore: Loading features from CSV files into FakeRedis...")
        import pandas as pd
        import os
        
        users_loaded = 0
        if os.path.exists("data/users.csv"):
            try:
                df = pd.read_csv("data/users.csv")
                for _, row in df.iterrows():
                    uid = int(row["user_id"])
                    pref_cats = str(row.get("preferred_categories", "")).split("|")
                    user_dict = {
                        "user_id": uid,
                        "age": int(row.get("age", 30)),
                        "age_bucket": int(row.get("age", 30)) // 10,
                        "gender": str(row.get("gender", "M")),
                        "country": str(row.get("country", "US")),
                        "country_id": 0,
                        "signup_days_ago": 30,
                        "num_interactions": 10,
                        "preferred_categories": pref_cats
                    }
                    self.store_user(uid, user_dict)
                    users_loaded += 1
            except Exception as e:
                logger.error(f"Error loading users CSV: {e}")
                
        # Fallback if no users loaded
        if users_loaded == 0:
            for uid in range(200):
                self.store_user(uid, {
                    "user_id": uid,
                    "age": 30,
                    "age_bucket": 3,
                    "gender": "M",
                    "country": "US",
                    "country_id": 0,
                    "signup_days_ago": 30,
                    "num_interactions": 10,
                    "preferred_categories": ["Music", "Gaming"]
                })
                
        items_loaded = 0
        if os.path.exists("data/videos.csv"):
            try:
                df = pd.read_csv("data/videos.csv")
                for _, row in df.iterrows():
                    iid = int(row["video_id"])
                    item_dict = {
                        "video_id": iid,
                        "item_id": iid,
                        "category": str(row.get("category", "Music")),
                        "category_id": 0,
                        "duration_seconds": int(row.get("duration_seconds", 300)),
                        "upload_days_ago": 10,
                        "like_ratio": 0.8,
                        "avg_watch_pct": 0.5,
                        "total_views": int(row.get("views_count", 100))
                    }
                    self.store_item(iid, item_dict)
                    items_loaded += 1
            except Exception as e:
                logger.error(f"Error loading videos CSV: {e}")
                
        # Fallback if no items loaded
        if items_loaded == 0:
            for iid in range(300):
                self.store_item(iid, {
                    "video_id": iid,
                    "item_id": iid,
                    "category": "Music",
                    "category_id": 0,
                    "duration_seconds": 300,
                    "upload_days_ago": 10,
                    "like_ratio": 0.8,
                    "avg_watch_pct": 0.5,
                    "total_views": 100
                })
                
        self.is_loaded = True
        logger.info("FeatureStore: Finished loading features into FakeRedis.")

    def get_user_features(self, user_id: int) -> dict:
        """Compatibility layer for legacy callers. Fetches user features from FakeRedis."""
        data = self.get_user(user_id)
        if data:
            # Map modern keys to legacy keys for compatibility
            if "age" not in data:
                data["age"] = data.get("age_bucket", 3) * 10 + 5
            if "country" not in data:
                data["country"] = str(data.get("country_id", 0))
            if "gender" not in data:
                data["gender"] = "M"
            return data
            
        return {
            "user_id": user_id,
            "age": 30,
            "gender": "M",
            "country": "US",
            "preferred_categories": ["Music", "Gaming"]
        }

    def get_item_features(self, video_id: int) -> dict:
        """Compatibility layer for legacy callers. Fetches item features from FakeRedis."""
        data = self.get_item(video_id)
        if data:
            if "category" not in data:
                data["category"] = str(data.get("category_id", 0))
            if "duration_seconds" not in data:
                data["duration_seconds"] = 300
            return data
            
        return {
            "video_id": video_id,
            "category": "Music",
            "duration_seconds": 300,
            "upload_days_ago": 10,
            "like_ratio": 0.8,
            "avg_watch_pct": 0.5,
            "total_views": 100
        }

# Backward-compatibility aliases and global singleton pattern
RealTimeFeatureStore = FeatureStore
feature_store = FeatureStore()
