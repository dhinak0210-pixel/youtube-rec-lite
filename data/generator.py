"""
Synthetic Data Generator for YouTube Recommendation System.

Generates highly realistic user profiles, catalog items, social graph connections,
and user-video interaction histories modeling YouTube-scale distributions.
Uses numpy power-laws, Pareto popularity curves, and preferential attachments.
Includes MockDataGenerator for full backend E2E backward compatibility.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from config import settings
from data.schemas import (
    InteractionType,
    UserProfile,
    ItemProfile,
    Interaction,
    UserSchema,
    VideoSchema,
    InteractionSchema
)

# ==========================================
# 🚀 Pydantic v2 Custom DataGenerator
# ==========================================

class DataGenerator:
    """Simulates realistic synthetic data for users, videos, graphs, and interactions."""

    def __init__(self, seed: int = 42):
        """Initializes the random number generator state."""
        self.rng = np.random.RandomState(seed)

    def generate_users(self, n: int = None) -> list[UserProfile]:
        """Generates sequential user profiles with power-law interaction capacities."""
        if n is None:
            n = settings.num_users

        logger.info(f"Generating {n} users...")
        users = []

        for uid in range(n):
            num_int = int(np.clip(self.rng.pareto(1.5) * 10, 0, 2000))
            
            # Select 1 to 4 distinct preferred categories from 0 to 19
            pref_count = self.rng.randint(1, 5)
            preferred = sorted(self.rng.choice(20, size=pref_count, replace=False).tolist())
            
            age_bucket = int(self.rng.randint(0, 7))
            gender = int(self.rng.randint(0, 3))
            country_id = int(self.rng.randint(0, 100))
            signup_days_ago = int(self.rng.randint(0, 1826))

            profile = UserProfile(
                user_id=uid,
                age_bucket=age_bucket,
                gender=gender,
                country_id=country_id,
                signup_days_ago=signup_days_ago,
                num_interactions=num_int,
                preferred_categories=preferred
            )
            users.append(profile)

        return users

    def generate_items(self, n: int = None) -> list[ItemProfile]:
        """Generates items modeling long-tail Pareto view popularities and video attributes."""
        if n is None:
            n = settings.num_items

        logger.info(f"Generating {n} items...")
        items = []

        for item_id in range(n):
            total_views = int(np.clip(self.rng.pareto(1.2) * 100, 0, 1000000))
            category_id = int(self.rng.randint(0, 20))
            duration_seconds = int(self.rng.choice([30, 60, 180, 300, 600, 1800]))
            upload_days_ago = int(self.rng.randint(0, 1001))
            
            # Creator range between 0 and n//5
            max_creator = max(1, n // 5)
            creator_id = int(self.rng.randint(0, max_creator + 1))
            
            like_ratio = round(float(self.rng.beta(5, 2)), 3)
            avg_watch_pct = round(float(self.rng.beta(2, 3)), 3)
            content_embedding = self.rng.randn(16).tolist()

            profile = ItemProfile(
                item_id=item_id,
                category_id=category_id,
                duration_seconds=duration_seconds,
                upload_days_ago=upload_days_ago,
                creator_id=creator_id,
                total_views=total_views,
                like_ratio=like_ratio,
                avg_watch_pct=avg_watch_pct,
                content_embedding=content_embedding
            )
            items.append(profile)

        return items

    def generate_interactions(
        self,
        users: list[UserProfile],
        items: list[ItemProfile],
        n: int = None
    ) -> list[Interaction]:
        """Simulates structured interactive feedback loops using category affinity boosts."""
        if n is None:
            n = settings.num_interactions

        logger.info(f"Generating {n} interactions...")
        
        # Build mapping of category ID to lists of item IDs
        category_to_items = {}
        for item in items:
            category_to_items.setdefault(item.category_id, []).append(item.item_id)

        # Build global item popularity weights
        item_views = np.array([item.total_views for item in items], dtype=float)
        views_sum = item_views.sum()
        if views_sum > 0:
            popularity_probs = item_views / views_sum
        else:
            popularity_probs = np.ones(len(items)) / len(items)

        item_ids = [item.item_id for item in items]
        interactions = []

        interaction_options = [
            InteractionType.VIEW,
            InteractionType.CLICK,
            InteractionType.LIKE,
            InteractionType.DISLIKE,
            InteractionType.SHARE,
            InteractionType.WATCH_COMPLETE
        ]
        interaction_weights = [0.35, 0.20, 0.15, 0.05, 0.10, 0.15]

        for i in range(n):
            # Pick random user
            user = users[self.rng.randint(0, len(users))]
            
            # Determine selection source
            if self.rng.rand() < 0.7 and user.preferred_categories:
                pref_cat = int(self.rng.choice(user.preferred_categories))
                cat_items = category_to_items.get(pref_cat, [])
                if cat_items:
                    item_id = int(self.rng.choice(cat_items))
                else:
                    item_id = int(self.rng.choice(item_ids, p=popularity_probs))
            else:
                item_id = int(self.rng.choice(item_ids, p=popularity_probs))

            # Draw interaction type by drawing index using rng.choice
            int_idx = self.rng.choice(len(interaction_options), p=interaction_weights)
            int_type = interaction_options[int_idx]

            # Determine watch ratio boundaries based on action type
            if int_type == InteractionType.WATCH_COMPLETE:
                watch_percentage = float(self.rng.uniform(0.9, 1.0))
            elif int_type in [InteractionType.LIKE, InteractionType.SHARE]:
                watch_percentage = float(self.rng.uniform(0.5, 1.0))
            elif int_type == InteractionType.DISLIKE:
                watch_percentage = float(self.rng.uniform(0.05, 0.3))
            else:
                watch_percentage = float(self.rng.beta(2, 3))

            watch_percentage = float(np.clip(watch_percentage, 0.0, 1.0))
            timestamp = 1700000000.0 + i * 10

            event = Interaction(
                user_id=user.user_id,
                item_id=item_id,
                interaction_type=int_type,
                timestamp=timestamp,
                watch_percentage=watch_percentage,
                context_hour=int(self.rng.randint(0, 24)),
                context_device=int(self.rng.randint(0, 4))
            )
            interactions.append(event)

        return sorted(interactions, key=lambda x: x.timestamp)

    def generate_social_graph(
        self,
        users: list[UserProfile],
        n_connections: int = None
    ) -> dict[int, list[int]]:
        """Maps user social networks using preferential attachment mechanics."""
        if n_connections is None:
            n_connections = settings.num_social_connections

        logger.info(f"Generating {n_connections} social connections...")
        
        social_graph = {user.user_id: [] for user in users}
        follower_counts = np.ones(len(users), dtype=float)

        for _ in range(n_connections):
            follower_id = int(self.rng.randint(0, len(users)))
            
            # Preferential followee choice (popular followees have higher prob)
            probs = follower_counts / follower_counts.sum()
            followee_id = int(self.rng.choice(len(users), p=probs))

            if follower_id != followee_id and followee_id not in social_graph[follower_id]:
                social_graph[follower_id].append(followee_id)
                follower_counts[followee_id] += 1.0

        return social_graph

    def generate_all(self) -> tuple[list[UserProfile], list[ItemProfile], list[Interaction], dict[int, list[int]]]:
        """Simulates all user demographics, video catalogs, click logs, and graphs simultaneously."""
        users = self.generate_users()
        items = self.generate_items()
        interactions = self.generate_interactions(users, items)
        social_graph = self.generate_social_graph(users)

        logger.info("--- Data Simulation Summary Stats ---")
        logger.info(f"Users generated      : {len(users)}")
        logger.info(f"Items generated      : {len(items)}")
        logger.info(f"Interactions generated: {len(interactions)}")
        logger.info(f"Social edges mapped  : {sum(len(followers) for followers in social_graph.values())}")
        logger.info("-------------------------------------")

        return users, items, interactions, social_graph


# ==========================================
# 🔄 Backward Compatibility Legacy Class
# ==========================================

class MockDataGenerator:
    """Legacy MockDataGenerator class to support existing codebase and fast fittings."""

    def __init__(self):
        # Apply standard random seeds for fully reproducible runs
        random.seed(settings.random_seed)
        np.random.seed(settings.random_seed)
        
        self.categories = [
            "Tech", "Music", "Gaming", "Comedy", "Education", 
            "Sports", "News", "Film", "Vlogs", "Cooking",
            "Science", "Travel", "Fashion", "Beauty", "Automotive",
            "Finance", "Fitness", "ASMR", "DIY", "Art"
        ][:settings.num_categories]
        
        self.countries = ["US", "IN", "BR", "CA", "GB", "DE", "FR", "JP", "AU", "ZA"]
        self.output_dir = "data"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_users(self) -> pd.DataFrame:
        """Generates demographic profiles and category affinity vectors."""
        users_list = []
        start_date = datetime.utcnow() - timedelta(days=365)
        
        for uid in range(settings.num_users):
            age = int(np.clip(np.random.normal(28, 8), 13, 85))
            gender = random.choice(["M", "F", "O"])
            country = random.choice(self.countries)
            
            # Select random number of category preferences
            pref_count = random.randint(1, 4)
            preferred = random.sample(self.categories, pref_count)
            signup_date = start_date + timedelta(seconds=random.randint(0, 31536000))
            
            # Validate row schema
            schema = UserSchema(
                user_id=uid, age=age, gender=gender, 
                country=country, preferred_categories=preferred,
                signup_date=signup_date
            )
            users_list.append({
                "user_id": schema.user_id,
                "age": schema.age,
                "gender": schema.gender,
                "country": schema.country,
                "preferred_categories": "|".join(schema.preferred_categories),
                "signup_date": schema.signup_date.isoformat()
            })
            
        df = pd.DataFrame(users_list)
        df.to_csv(os.path.join(self.output_dir, "users.csv"), index=False)
        return df

    def generate_videos(self) -> pd.DataFrame:
        """Generates video catalogs modeling long-tail video metadata."""
        videos_list = []
        start_date = datetime.utcnow() - timedelta(days=180)
        
        for vid in range(settings.num_items):
            category = random.choice(self.categories)
            # Add randomized tags
            tags = [category.lower(), f"tag_{random.randint(1, 100)}"]
            duration = int(np.clip(np.random.exponential(360) + 30, 10, 7200)) # mean ~6 min
            upload_date = start_date + timedelta(seconds=random.randint(0, 15552000))
            
            schema = VideoSchema(
                video_id=vid, category=category, tags=tags,
                duration_seconds=duration, upload_date=upload_date
            )
            videos_list.append({
                "video_id": schema.video_id,
                "category": schema.category,
                "tags": "|".join(schema.tags),
                "duration_seconds": schema.duration_seconds,
                "upload_date": schema.upload_date.isoformat()
            })
            
        df = pd.DataFrame(videos_list)
        df.to_csv(os.path.join(self.output_dir, "videos.csv"), index=False)
        return df

    def generate_interactions(self, users_df: pd.DataFrame, videos_df: pd.DataFrame) -> pd.DataFrame:
        """Generates structured interactions using a Zipfian/Pareto power-law distribution."""
        interactions_list = []
        start_date = datetime.utcnow() - timedelta(days=30)
        
        # Modeling long-tail power law item popularity weights
        weights = 1.0 / (np.arange(1, settings.num_items + 1) ** 0.8)
        weights /= weights.sum()
        
        # Build mapping of video categories
        vid_categories = dict(zip(videos_df["video_id"], videos_df["category"]))
        user_affinities = {}
        for _, row in users_df.iterrows():
            user_affinities[row["user_id"]] = row["preferred_categories"].split("|")
            
        # Draw interaction candidates
        selected_vids = np.random.choice(videos_df["video_id"], size=settings.num_interactions, p=weights)
        
        for idx in range(settings.num_interactions):
            uid = random.randint(0, settings.num_users - 1)
            vid = int(selected_vids[idx])
            
            # Formulate user category affinity context boost
            affinity_match = vid_categories[vid] in user_affinities[uid]
            base_click_prob = 0.45 if affinity_match else 0.15
            click = 1 if random.random() < base_click_prob else 0
            
            watch_ratio = 0.0
            like = 0
            dislike = 0
            share = 0
            
            if click == 1:
                # Clicked items trigger exponential watch distribution
                watch_ratio = float(np.clip(np.random.beta(2, 2 if affinity_match else 5), 0.0, 1.0))
                if watch_ratio > 0.6:
                    like = 1 if random.random() < 0.4 else 0
                    share = 1 if random.random() < 0.1 else 0
                elif watch_ratio < 0.15:
                    dislike = 1 if random.random() < 0.15 else 0
                    
            timestamp = start_date + timedelta(seconds=random.randint(0, 2592000))
            
            schema = InteractionSchema(
                user_id=uid, video_id=vid, timestamp=timestamp,
                click=click, watch_ratio=watch_ratio,
                like=like, dislike=dislike, share=share
            )
            interactions_list.append({
                "user_id": schema.user_id,
                "video_id": schema.video_id,
                "timestamp": schema.timestamp.isoformat(),
                "click": schema.click,
                "watch_ratio": schema.watch_ratio,
                "like": schema.like,
                "dislike": schema.dislike,
                "share": schema.share
            })
            
        df = pd.DataFrame(interactions_list)
        df.to_csv(os.path.join(self.output_dir, "interactions.csv"), index=False)
        return df

    def generate_all(self):
        """Generates all datasets simultaneously."""
        print("🌱 Generating synthetic users...")
        users = self.generate_users()
        print("🌱 Generating synthetic videos...")
        videos = self.generate_videos()
        print("🌱 Simulating interactions (applying Zipfian distribution)...")
        self.generate_interactions(users, videos)
        print("✅ Dataset generation complete! All CSV assets saved to /data.")
