"""
Cold Start Recommendation Engine containing:
1. Global/Category Popularity Model: Fallback for entirely new or unprofiled users.
2. Content Similarity Model: Similarity-based seed matching using unit-normalized embeddings.
3. LinUCB Contextual Bandit Model: Smart exploration-exploitation balance for profiled cold users.
4. ColdStartHandler: Coordinates popular candidate generation and bandit scoring.
5. ColdStartRecommender: High-fidelity backward-compatible adapter that fits on legacy CSV tables or Pydantic lists.
"""

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional, Set, Tuple
from loguru import logger
from data.schemas import UserProfile, ItemProfile, ScoredItem, Interaction, InteractionType

class PopularityModel:
    """
    Computes global and category-specific popularity scores based on interaction weights.
    Direct category matches receive full normalized popularity, while global catalog items
    are served with a 0.5x discount multiplier.
    """
    def __init__(self):
        self.global_pop: Dict[int, float] = {}  # item_id -> normalized score [0, 1]
        self.category_pop: Dict[int, Dict[int, float]] = defaultdict(dict)  # cat -> {item_id -> score}
        self.item_to_cat: Dict[int, int] = {}  # item_id -> category_id
        self._fitted = False

    def fit(self, interactions: List[Interaction], items: List[ItemProfile]) -> "PopularityModel":
        """
        Calculates weighted popularity score per item, normalizes scores to [0, 1],
        and maps category-level indices.
        """
        logger.info("Fitting weighted PopularityModel metrics...")
        
        # 1. Compile map of items to categories
        self.item_to_cat = {item.item_id: item.category_id for item in items}
        cat_to_items = defaultdict(list)
        for item in items:
            cat_to_items[item.category_id].append(item.item_id)

        # 2. Sum weighted interactions per item
        item_scores = defaultdict(float)
        for inter in interactions:
            item_scores[inter.item_id] += inter.weight

        # Ensure all catalog items are initialized in the scoring dictionary
        for item in items:
            if item.item_id not in item_scores:
                item_scores[item.item_id] = 0.0

        # 3. Global normalization to [0, 1]
        max_global_score = max(item_scores.values()) if item_scores else 0.0
        self.global_pop = {}
        for item_id, score in item_scores.items():
            self.global_pop[item_id] = score / max_global_score if max_global_score > 0 else 0.0

        # 4. Category-level normalization to [0, 1]
        self.category_pop = defaultdict(dict)
        for cat, item_ids in cat_to_items.items():
            cat_scores = {item_id: item_scores[item_id] for item_id in item_ids}
            max_cat_score = max(cat_scores.values()) if cat_scores else 0.0
            for item_id, score in cat_scores.items():
                self.category_pop[cat][item_id] = score / max_cat_score if max_cat_score > 0 else 0.0

        self._fitted = True
        logger.info(f"PopularityModel indexed {len(self.global_pop)} items across {len(self.category_pop)} categories.")
        return self

    def recommend(self, user: Optional[UserProfile] = None, n: int = 100, exclude: Optional[Set[int]] = None) -> List[ScoredItem]:
        """
        Retrieves category-boosted or global popular candidates.
        """
        exclude_set = exclude if exclude is not None else set()
        scores = {}

        if user and user.preferred_categories:
            for item_id, g_score in self.global_pop.items():
                if item_id in exclude_set:
                    continue
                item_cat = self.item_to_cat.get(item_id)
                if item_cat in user.preferred_categories:
                    # Category match gets full category-level popularity score
                    score = self.category_pop[item_cat].get(item_id, g_score)
                else:
                    # Non-preferred category items get a 0.5x global discount
                    score = g_score * 0.5
                scores[item_id] = score
        else:
            # Fallback to plain global popularity for entirely new/anonymous requests
            for item_id, g_score in self.global_pop.items():
                if item_id in exclude_set:
                    continue
                scores[item_id] = g_score

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            ScoredItem(item_id=item_id, score=float(score), source="popularity") 
            for item_id, score in sorted_items
        ]

class ContentModel:
    """
    Computes cosine similarity between units of catalog assets using
    normalized semantic content embeddings.
    """
    def __init__(self):
        self.item_vecs: Dict[int, np.ndarray] = {}  # item_id -> unit-normalized vector

    def fit(self, items: List[ItemProfile]) -> "ContentModel":
        """
        Extracts and normalizes unit content embeddings from catalog items.
        """
        logger.info("Fitting content similarity vectors...")
        count = 0
        for item in items:
            if item.content_embedding:
                vec = np.array(item.content_embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                self.item_vecs[item.item_id] = vec / norm if norm > 0 else vec
                count += 1
        
        logger.info(f"ContentModel indexed {count} items with content embeddings.")
        return self

    def similar(self, seeds: List[int], n: int = 50, exclude: Optional[Set[int]] = None) -> List[ScoredItem]:
        """
        Averages seed vectors and scores all items based on cosine similarity.
        """
        exclude_set = exclude if exclude is not None else set()
        seed_set = set(seeds)

        # Average candidate seed embeddings
        seed_vectors = [self.item_vecs[s] for s in seeds if s in self.item_vecs]
        if not seed_vectors:
            return []
        
        avg_vector = np.mean(seed_vectors, axis=0)
        avg_norm = np.linalg.norm(avg_vector)
        avg_vector = avg_vector / avg_norm if avg_norm > 0 else avg_vector

        # Calculate cosine similarity dot product (vectors are normalized)
        scores = {}
        for item_id, vec in self.item_vecs.items():
            if item_id in exclude_set or item_id in seed_set:
                continue
            scores[item_id] = float(np.dot(avg_vector, vec))

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            ScoredItem(item_id=item_id, score=float(sim), source="content") 
            for item_id, sim in sorted_items
        ]

class LinUCBBandit:
    """
    Linear Upper Confidence Bound (LinUCB) Contextual Multi-Armed Bandit algorithm.
    Balances exploitation (expected reward based on demographics) and exploration
    (uncertainty boundaries) for cold start user routing.
    """
    def __init__(self, ctx_dim: int = 8, alpha: float = 1.0):
        self.alpha = alpha
        self.ctx_dim = ctx_dim
        self.A: Dict[int, np.ndarray] = {}  # item_id -> (ctx_dim x ctx_dim) covariance matrix
        self.b: Dict[int, np.ndarray] = {}  # item_id -> (ctx_dim,) reward tracker vector

    def _init_item(self, item_id: int):
        """Initializes covariance matrix A to Identity and vector b to zero if unseen."""
        if item_id not in self.A:
            self.A[item_id] = np.identity(self.ctx_dim, dtype=np.float32)
            self.b[item_id] = np.zeros(self.ctx_dim, dtype=np.float32)

    def context(self, user: UserProfile) -> np.ndarray:
        """
        Compiles user demographics into normalized 8-dimensional bandit feature vector.
        """
        return np.array([
            user.age_bucket / 6.0,
            user.gender / 2.0,
            user.country_id / 99.0,
            min(user.signup_days_ago / 365.0, 10.0) / 10.0,
            len(user.preferred_categories) / 5.0,
            1.0 if user.is_cold_start else 0.0,
            0.5,  # Placeholder 1
            0.5   # Placeholder 2
        ], dtype=np.float32)

    def select(self, user: UserProfile, candidates: List[int], n: int = 20) -> List[ScoredItem]:
        """
        Calculates UCB scores for all candidate items and selects the top n arms.
        """
        ctx = self.context(user)
        scores = []

        for item in candidates:
            self._init_item(item)
            A_inv = np.linalg.inv(self.A[item])
            theta = A_inv @ self.b[item]
            
            exploit = float(ctx @ theta)
            explore = float(self.alpha * np.sqrt(ctx @ A_inv @ ctx))
            
            ucb_score = exploit + explore
            scores.append((item, ucb_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            ScoredItem(item_id=item_id, score=float(score), source="bandit") 
            for item_id, score in scores[:n]
        ]

    def update(self, item_id: int, user: UserProfile, reward: float):
        """Updates covariance A and vector b with observed interaction rewards."""
        ctx = self.context(user)
        self._init_item(item_id)
        self.A[item_id] += np.outer(ctx, ctx)
        self.b[item_id] += reward * ctx
        logger.debug(f"Updated LinUCB arm for video {item_id} with reward {reward}.")

class ColdStartHandler:
    """
    Coordinating handler orchestrating popularity modeling, semantic matching,
    and online bandit exploration for cold environments.
    """
    def __init__(self):
        self.popularity = PopularityModel()
        self.content = ContentModel()
        self.bandit = LinUCBBandit()

    def fit(self, interactions: List[Interaction], items: List[ItemProfile]) -> "ColdStartHandler":
        """Precomputes offline structures and builds catalog vectors."""
        self.popularity.fit(interactions, items)
        self.content.fit(items)
        return self

    def recommend(self, user: UserProfile, n: int = 100, exclude: Optional[List[int]] = None) -> List[ScoredItem]:
        """
        Retrieves UCB bandit recommendations from the top 2x popular items candidate pool.
        """
        exclude_set = set(exclude) if exclude is not None else set()
        
        # 1. Retrieve 2x popular items for active cohort
        popular_candidates = self.popularity.recommend(user=user, n=2 * n, exclude=exclude_set)
        candidate_ids = [item.item_id for item in popular_candidates]
        
        if not candidate_ids:
            return []
            
        # 2. Select and rank candidates using contextual Multi-Armed Bandit UCB bounds
        return self.bandit.select(user=user, candidates=candidate_ids, n=n)

class ColdStartRecommender:
    """
    Backward-compatible coordinator adapter that matches the central ModelRegistry lifecycle.
    Seamlessly supports both Pydantic lists and legacy CSV tables.
    """
    def __init__(self):
        self.handler = ColdStartHandler()
        self.user_profiles: Dict[int, UserProfile] = {}
        
        # Default cold threshold value from config
        try:
            from config import settings
            self.threshold = settings.cold_start_threshold
        except Exception:
            self.threshold = 5

    def fit(self, users: Optional[List[UserProfile]] = None, items: Optional[List[ItemProfile]] = None, 
            interactions: Optional[List[Interaction]] = None, data_dir: str = "data"):
        """
        Trains the models. Gracefully parses CSV tables if Pydantic lists are omitted.
        """
        logger.info("Initializing ColdStartRecommender fitting lifecycle...")
        
        parsed_users: List[UserProfile] = []
        parsed_items: List[ItemProfile] = []
        parsed_interactions: List[Interaction] = []

        if users is not None and items is not None and interactions is not None:
            parsed_users = users
            parsed_items = items
            parsed_interactions = interactions
        else:
            # Load legacy dataset tables from CSV
            users_path = os.path.join(data_dir, "users.csv")
            videos_path = os.path.join(data_dir, "videos.csv")
            interactions_path = os.path.join(data_dir, "interactions.csv")
            
            if not (os.path.exists(users_path) and os.path.exists(videos_path) and os.path.exists(interactions_path)):
                logger.error(f"Target CSV files not found in {data_dir}. Skipping parsing.")
                return self
                
            users_df = pd.read_csv(users_path)
            videos_df = pd.read_csv(videos_path)
            interactions_df = pd.read_csv(interactions_path)

            # Establish category list maps
            unique_cats = sorted(list(videos_df["category"].unique()))
            cat_to_id = {cat: idx for idx, cat in enumerate(unique_cats)}

            interaction_counts = interactions_df.groupby("user_id").size().to_dict()
            gender_map = {"M": 0, "F": 1, "O": 2}
            
            unique_countries = sorted(list(users_df["country"].unique()))
            country_to_id = {c: min(idx, 99) for idx, c in enumerate(unique_countries)}
            
            # Map legacy UserSchema to Pydantic UserProfile
            for _, row in users_df.iterrows():
                uid = int(row["user_id"])
                age = int(row["age"])
                age_bucket = int(np.clip((age - 13) // 10, 0, 6))
                gender = gender_map.get(str(row["gender"]), 2)
                country_id = country_to_id.get(str(row["country"]), 0)
                
                try:
                    signup_dt = pd.to_datetime(row["signup_date"])
                    signup_days = max(0, int((pd.Timestamp.utcnow() - signup_dt.tz_localize(None)).days))
                except Exception:
                    signup_days = 30
                    
                pref_str = str(row["preferred_categories"])
                preferred = [cat_to_id[p] for p in pref_str.split("|") if p in cat_to_id]
                num_int = interaction_counts.get(uid, 0)
                
                parsed_users.append(UserProfile(
                    user_id=uid,
                    age_bucket=age_bucket,
                    gender=gender,
                    country_id=country_id,
                    signup_days_ago=signup_days,
                    num_interactions=num_int,
                    preferred_categories=preferred
                ))

            # Pre-aggregate view and like metrics per video
            views_dict = interactions_df[interactions_df["click"] == 1].groupby("video_id").size().to_dict()
            likes_dict = interactions_df.groupby("video_id")["like"].sum().to_dict()

            # Map legacy VideoSchema to Pydantic ItemProfile
            for _, row in videos_df.iterrows():
                vid = int(row["video_id"])
                cat_id = cat_to_id.get(str(row["category"]), 0)
                dur = int(row["duration_seconds"])
                
                try:
                    upload_dt = pd.to_datetime(row["upload_date"])
                    upload_days = max(0, int((pd.Timestamp.utcnow() - upload_dt.tz_localize(None)).days))
                except Exception:
                    upload_days = 10
                    
                views = views_dict.get(vid, 0)
                likes = likes_dict.get(vid, 0)
                like_ratio = min(max(float(likes / max(views, 1)), 0.0), 1.0)
                
                vid_inters = interactions_df[interactions_df["video_id"] == vid]
                avg_watch = float(vid_inters["watch_ratio"].mean()) if len(vid_inters) > 0 else 0.5
                
                # Content embeddings (reproducible randomized vector placeholder)
                rng = np.random.RandomState(vid)
                emb = rng.randn(16).tolist()
                
                parsed_items.append(ItemProfile(
                    item_id=vid,
                    category_id=cat_id,
                    duration_seconds=dur,
                    upload_days_ago=upload_days,
                    creator_id=0,
                    total_views=views,
                    like_ratio=like_ratio,
                    avg_watch_pct=avg_watch,
                    content_embedding=emb
                ))

            # Map legacy InteractionSchema to Pydantic Interaction
            for _, row in interactions_df.iterrows():
                uid = int(row["user_id"])
                vid = int(row["video_id"])
                click = int(row["click"])
                watch = float(row["watch_ratio"])
                like = int(row["like"])
                dislike = int(row["dislike"])
                share = int(row["share"])
                
                if share == 1:
                    t = InteractionType.SHARE
                elif like == 1:
                    t = InteractionType.LIKE
                elif dislike == 1:
                    t = InteractionType.DISLIKE
                elif watch >= 0.9:
                    t = InteractionType.WATCH_COMPLETE
                elif click == 1:
                    t = InteractionType.CLICK
                else:
                    t = InteractionType.VIEW
                    
                try:
                    ts = pd.to_datetime(row["timestamp"]).timestamp()
                except Exception:
                    ts = float(time.time())
                    
                parsed_interactions.append(Interaction(
                    user_id=uid,
                    item_id=vid,
                    interaction_type=t,
                    timestamp=ts,
                    watch_percentage=watch,
                    context_hour=12,
                    context_device=0
                ))

        # Fit core handler components
        self.user_profiles = {u.user_id: u for u in parsed_users}
        self.handler.fit(parsed_interactions, parsed_items)
        logger.info("ColdStartRecommender fit completed.")
        return self

    def is_cold_user(self, user_id: int) -> bool:
        """Determines if a user profile is cold based on interaction thresholds."""
        profile = self.user_profiles.get(user_id)
        if profile is not None:
            return profile.is_cold_start
        return True

    def recommend(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieves recommended candidates as scored tuples (item_id, score) for compatibility.
        """
        profile = self.user_profiles.get(user_id)
        if profile is None:
            # Spawn basic default user profile
            profile = UserProfile(
                user_id=user_id,
                age_bucket=3,
                gender=0,
                country_id=0,
                signup_days_ago=30,
                num_interactions=0,
                preferred_categories=[]
            )
            
        scored_items = self.handler.recommend(user=profile, n=top_n)
        return [(item.item_id, item.score) for item in scored_items]
