import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path
from src.config import RAW_DATA_DIR
from src.utils.logger import logger

class YouTubeSyntheticDataGenerator:
    """
    Production-grade synthetic dataset generator simulating authentic YouTube ecosystems:
    - User profiles with personalized niche preferences.
    - Heavy-tailed interaction profiles and scale-free social follow graphs (Power-Law / Pareto).
    - Long-tail video popularity distributions (viral hits vs. long-tail niche clips).
    - Session engagement telemetry (click, watch percentage, like, dislike, share, complete) 
      correlated through dynamic joint probabilities.
    - Temporal interaction timestamps allowing structural train/test chronological splitting.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.set_seed(seed)
        
        # 20 realistic YouTube categories
        self.categories = [
            "Music", "Gaming", "Vlogs", "Tech", "Comedy", 
            "Education", "News", "Sports", "Beauty", "Cooking", 
            "DIY", "Travel", "Movies", "Anime", "ASMR", 
            "Fitness", "History", "Science", "Finance", "Pets"
        ]
        
        self.countries = ["US", "IN", "BR", "MX", "DE", "GB", "FR", "JP", "KR", "CA"]
        self.genders = ["M", "F", "O"]

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def generate_users(self, num_users: int = 10000) -> pd.DataFrame:
        """
        Generates user profiles with realistic demographic dimensions, sign-up times, 
        niche interests, and interaction budgets following a Pareto power-law.
        
        Realistic Pattern:
        - Category preference: Users choose 1 to 4 categories representing personalized taste clusters.
        - Interaction budget: User activity levels are extremely skewed—most users watch a few videos,
          while a tiny minority (hyper-active users) consume thousands.
        """
        logger.info(f"Generating {num_users} realistic user profiles...")
        
        user_ids = np.arange(num_users)
        ages = np.random.randint(13, 75, size=num_users)
        genders = np.random.choice(self.genders, size=num_users, p=[0.48, 0.48, 0.04])
        countries = np.random.choice(self.countries, size=num_users, p=[0.25, 0.20, 0.12, 0.08, 0.07, 0.07, 0.06, 0.05, 0.05, 0.05])
        
        # Signups distributed over the last 2 years (730 days)
        today = pd.Timestamp.now()
        days_offset = np.random.randint(1, 730, size=num_users)
        signup_dates = [today - pd.to_timedelta(days, unit='d') for days in days_offset]
        
        # Determine preferred categories (1 to 4 categories per user)
        preferred_cats_list = []
        for _ in range(num_users):
            n_cats = random.randint(1, 4)
            chosen_cats = random.sample(self.categories, n_cats)
            preferred_cats_list.append("|".join(chosen_cats))
            
        # Generate activity budget scale following Pareto distribution (a=1.5 is highly skewed)
        # Yields a heavy-tailed power-law distribution
        activity_budget = (np.random.pareto(a=1.5, size=num_users) + 1.0)
        # Normalize to represent weights for sample distributions
        activity_weight = activity_budget / activity_budget.sum()
        
        user_df = pd.DataFrame({
            "user_id": user_ids,
            "age": ages,
            "gender": genders,
            "country": countries,
            "signup_date": signup_dates,
            "preferred_categories": preferred_cats_list,
            "activity_weight": activity_weight
        })
        
        logger.info("User generation complete.")
        return user_df

    def generate_videos(self, num_videos: int = 50000) -> pd.DataFrame:
        """
        Generates video metadata including semantic embeddings and Pareto view-distributions.
        
        Realistic Pattern:
        - View counts: Skewed heavily towards a few viral videos (e.g. popular music/gaming uploads) 
          and a long tail of items with < 5 views.
        - Embeddings: 32-dim unit-normalized vectors representing deep semantic features.
        - Likability & Watchability: Associated with video quality indices and upload age.
        """
        logger.info(f"Generating {num_videos} video assets...")
        
        video_ids = np.arange(num_videos)
        categories = np.random.choice(self.categories, size=num_videos)
        
        # Creators (1,500 distinct creators representing influencers/channels)
        creators = [f"channel_{i}" for i in np.random.randint(0, 1500, size=num_videos)]
        
        # Durations in seconds (Log-Normal: many short 1-3 min clips, few 1-hour logs)
        durations = np.random.lognormal(mean=5.5, sigma=1.0, size=num_videos).astype(int)
        # Clamp to [15, 7200] range
        durations = np.clip(durations, 15, 7200)
        
        # Upload dates distributed over last 1 year (365 days)
        today = pd.Timestamp.now()
        days_offset = np.random.randint(0, 365, size=num_videos)
        upload_dates = [today - pd.to_timedelta(days, unit='d') for days in days_offset]
        
        # Embeddings: L2 unit-normalized random vectors
        raw_embeds = np.random.normal(size=(num_videos, 32))
        norms = np.linalg.norm(raw_embeds, axis=1, keepdims=True)
        embeddings = raw_embeds / norms
        embeddings_list = [embed.tolist() for embed in embeddings]
        
        # Pre-calculate a quality/appeal score representing intrinsic likability
        quality_scores = np.random.beta(a=2.0, b=5.0, size=num_videos)
        
        # Pre-calculate baseline popularity weights (Pareto a=1.2: few extreme outliers)
        popularity_budget = np.random.pareto(a=1.2, size=num_videos) + 1.0
        popularity_weight = popularity_budget / popularity_budget.sum()
        
        video_df = pd.DataFrame({
            "video_id": video_ids,
            "category": categories,
            "duration": durations,
            "upload_date": upload_dates,
            "creator": creators,
            "quality_score": quality_scores,
            "popularity_weight": popularity_weight,
            "content_embedding": embeddings_list
        })
        
        logger.info("Video generation complete.")
        return video_df

    def generate_interactions(self, 
                              users_df: pd.DataFrame, 
                              videos_df: pd.DataFrame, 
                              num_interactions: int = 500000) -> pd.DataFrame:
        """
        Simulates interaction telemetry linking users and videos.
        
        Realistic Pattern:
        - 70% personalized traffic: users pull items belonging to their listed category preferences.
        - 30% exploration traffic: users watch globally trending/popular items (independent of preference).
        - Multi-objective outcomes: watch ratio, click, like, dislike, share, complete are intercorrelated:
          * Clicks are highly likely for preferred categories.
          * Watch ratio depends on user-item match and video quality score.
          * Shares and likes only occur if watch ratio is very high.
          * Dislikes occur if the watch ratio is low (e.g. clickbait).
        """
        logger.info(f"Generating {num_interactions} interaction telemetry rows (vectorized execution)...")
        today = pd.Timestamp.now()
        
        # 1. Sample user indices for all interactions based on their power-law activity weights
        sampled_user_idxs = np.random.choice(
            users_df["user_id"].values, 
            size=num_interactions, 
            p=users_df["activity_weight"].values
        )
        
        # Map user_id to lists of preferred categories for fast checks
        user_to_pref = dict(zip(users_df["user_id"], users_df["preferred_categories"].str.split("|")))
        
        # Cache video indices sorted by categories for personalized lookups
        video_cat_groups: Dict[str, np.ndarray] = {}
        video_cat_weights: Dict[str, np.ndarray] = {}
        for cat in self.categories:
            cat_vids = videos_df[videos_df["category"] == cat]
            if len(cat_vids) > 0:
                video_cat_groups[cat] = cat_vids["video_id"].values
                # Normalize category-specific popularity weights
                weights = cat_vids["popularity_weight"].values
                video_cat_weights[cat] = weights / weights.sum()
                
        # 2. Decide Personalized (70%) vs Exploration (30%) splits
        is_personalized = np.random.rand(num_interactions) < 0.7
        
        # Pre-sample globally popular videos for the 30% exploratory interactions
        glob_vids = videos_df["video_id"].values
        glob_weights = videos_df["popularity_weight"].values
        # Ensure exact probability sum
        glob_weights = glob_weights / glob_weights.sum()
        
        exploratory_vids = np.random.choice(glob_vids, size=num_interactions, p=glob_weights)
        
        # 3. Vectorized and batched item selection
        sampled_video_ids = np.zeros(num_interactions, dtype=int)
        
        # Exploration direct assignment
        sampled_video_ids[~is_personalized] = exploratory_vids[~is_personalized]
        
        # Personalization (70%): Map to preferred categories
        logger.info("Mapping personalized interactions to user interest clusters...")
        for i in range(num_interactions):
            if is_personalized[i]:
                uid = sampled_user_idxs[i]
                pref_cats = user_to_pref[uid]
                
                # Choose one preferred category randomly
                chosen_cat = random.choice(pref_cats)
                
                # Fetch category video pool
                cat_pool = video_cat_groups.get(chosen_cat)
                if cat_pool is not None:
                    cat_w = video_cat_weights[chosen_cat]
                    # Sample a video within category
                    # To keep it fast, we can sample a single item quickly
                    idx = np.random.choice(len(cat_pool), p=cat_w)
                    sampled_video_ids[i] = cat_pool[idx]
                else:
                    # Fallback to exploration video if category empty
                    sampled_video_ids[i] = exploratory_vids[i]

        # 4. Construct telemetry DataFrame and generate joint probabilities
        logger.info("Evaluating multi-objective click and engagement outcomes...")
        interactions_df = pd.DataFrame({
            "user_id": sampled_user_idxs,
            "video_id": sampled_video_ids,
            "is_personalized": is_personalized
        })
        
        # Fetch metadata mappings for calculations
        video_lookup = videos_df.set_index("video_id")
        
        # Extract features for all interactions
        duration_arr = video_lookup.loc[sampled_video_ids, "duration"].values
        quality_arr = video_lookup.loc[sampled_video_ids, "quality_score"].values
        category_arr = video_lookup.loc[sampled_video_ids, "category"].values
        
        # Click probability calculations: Higher click-through rate (CTR) if personalized
        # Adding random Gumbel noise to simulate decisions
        ctr_base = np.where(is_personalized, 0.45, 0.12)
        click_logits = ctr_base + 0.15 * quality_arr + np.random.normal(0, 0.1, size=num_interactions)
        clicks = (click_logits > 0.35).astype(int)
        
        # Watch ratio: Dependent on quality score and personalized match
        # Fits a Beta distribution using clicks:
        # If click=0, watch ratio is tiny (e.g. scroll past). 
        # If click=1, watch ratio is modeled based on video appeal.
        watch_ratios = np.zeros(num_interactions)
        
        # Segment indices for clicked vs unclicked
        clicked_indices = np.where(clicks == 1)[0]
        unclicked_indices = np.where(clicks == 0)[0]
        
        # Unclicked: watched between 0% and 5% (e.g., auto-play hover)
        watch_ratios[unclicked_indices] = np.random.uniform(0.0, 0.05, size=len(unclicked_indices))
        
        # Clicked: watch ratio is modeled as Beta distribution dependent on quality
        if len(clicked_indices) > 0:
            alpha = 2.0 + 4.0 * quality_arr[clicked_indices]
            beta = 1.5 + 2.0 * (1.0 - quality_arr[clicked_indices])
            watch_ratios[clicked_indices] = np.random.beta(alpha, beta)
            
        # Interaction events correlated with watch ratios:
        # - watch_complete: 1 if watch_ratio >= 0.90
        # - like: occurs if clicked and watch ratio is high
        # - dislike: occurs if clicked and watch ratio is low (clickbait index)
        # - share: occurs if clicked and watch ratio is extremely high (> 80%)
        watch_complete = (watch_ratios >= 0.90).astype(int)
        
        likes = np.zeros(num_interactions, dtype=int)
        dislikes = np.zeros(num_interactions, dtype=int)
        shares = np.zeros(num_interactions, dtype=int)
        
        if len(clicked_indices) > 0:
            w_clicked = watch_ratios[clicked_indices]
            q_clicked = quality_arr[clicked_indices]
            
            # Like probability = watch_ratio^2 * quality
            like_prob = (w_clicked ** 1.8) * q_clicked
            likes[clicked_indices] = (np.random.rand(len(clicked_indices)) < like_prob).astype(int)
            
            # Dislike probability = (1 - watch_ratio)^2 * (1 - quality)
            dislike_prob = ((1.0 - w_clicked) ** 2.0) * (1.0 - q_clicked)
            dislikes[clicked_indices] = (np.random.rand(len(clicked_indices)) < dislike_prob).astype(int)
            
            # Share probability = watch_ratio^3 * quality * 0.4
            share_prob = (w_clicked ** 3.0) * q_clicked * 0.4
            shares[clicked_indices] = (np.random.rand(len(clicked_indices)) < share_prob).astype(int)
            
        # Ensure likes and dislikes are mutually exclusive
        dislikes[likes == 1] = 0
        
        # 5. Temporal stamps distributed over the last 30 days
        days_ago = np.random.uniform(0, 30, size=num_interactions)
        timestamps = today - pd.to_timedelta(days_ago, unit='d')
        
        # Merge all telemetry columns
        interactions_df["timestamp"] = timestamps
        interactions_df["click"] = clicks
        interactions_df["watch_percentage"] = watch_ratios * 100.0
        interactions_df["watch_ratio"] = watch_ratios
        interactions_df["watch_time_seconds"] = (watch_ratios * duration_arr).astype(int)
        interactions_df["like"] = likes
        interactions_df["dislike"] = dislikes
        interactions_df["share"] = shares
        interactions_df["watch_complete"] = watch_complete
        
        # Sort chronologically to support clean temporal partitions (train/test split)
        interactions_df = interactions_df.sort_values(by="timestamp").reset_index(drop=True)
        
        logger.info("Interaction telemetry complete.")
        return interactions_df

    def generate_social_graph(self, num_users: int = 10000, num_connections: int = 50000) -> pd.DataFrame:
        """
        Generatesfollow connections among users.
        
        Realistic Pattern (Scale-free network / Preferential attachment):
        - Follow connections follow a power-law degree distribution.
        - A small handful of users (e.g. active creators or influencers) have thousands of followers,
          while the vast majority have very few follows.
        """
        logger.info(f"Generating scale-free social follow graph with {num_connections} edges...")
        
        # Generate recipient popularity budgets using a Pareto distribution (a=1.3: very skewed)
        pop_budget = np.random.pareto(a=1.3, size=num_users) + 1.0
        pop_weight = pop_budget / pop_budget.sum()
        
        # Sample followee_ids (influencers being followed) based on these weights
        followee_ids = np.random.choice(np.arange(num_users), size=num_connections, p=pop_weight)
        
        # Sample follower_ids randomly
        follower_ids = np.random.choice(np.arange(num_users), size=num_connections)
        
        # Remove self-follows
        self_follows = follower_ids == followee_ids
        while self_follows.any():
            follower_ids[self_follows] = np.random.choice(np.arange(num_users), size=self_follows.sum())
            self_follows = follower_ids == followee_ids
            
        social_df = pd.DataFrame({
            "follower_id": follower_ids,
            "followee_id": followee_ids
        })
        
        # Deduplicate follow relationships
        social_df = social_df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Social graph generated. Unique follow connections: {len(social_df)}")
        return social_df

    def generate_all(self, 
                     num_users: int = 10000, 
                     num_videos: int = 50000, 
                     num_interactions: int = 500000, 
                     num_follows: int = 50000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes and saves the entire YouTube synthetic data generator.
        """
        logger.info("Executing comprehensive YouTube Synthetic Data Generator...")
        
        users_df = self.generate_users(num_users)
        videos_df = self.generate_videos(num_videos)
        interactions_df = self.generate_interactions(users_df, videos_df, num_interactions)
        social_df = self.generate_social_graph(num_users, num_follows)
        
        # Save raw datasets to raw directory
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        users_df.to_csv(RAW_DATA_DIR / "users.csv", index=False)
        videos_df.to_csv(RAW_DATA_DIR / "videos.csv", index=False)
        interactions_df.to_csv(RAW_DATA_DIR / "interactions.csv", index=False)
        social_df.to_csv(RAW_DATA_DIR / "social_graph.csv", index=False)
        
        logger.info(f"All synthetic datasets generated and saved to {RAW_DATA_DIR}.")
        return users_df, videos_df, interactions_df, social_df

# Standalone loader function matching codebase signature
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Integrates the custom generator seamlessly into the existing project pipelines.
    Loads or generates user, video, and interaction frames.
    """
    user_path = RAW_DATA_DIR / "users.csv"
    video_path = RAW_DATA_DIR / "videos.csv"
    interaction_path = RAW_DATA_DIR / "interactions.csv"
    
    if not (user_path.exists() and video_path.exists() and interaction_path.exists()):
        logger.info("Target datasets not found. Triggering the YouTube Power-Law Generator.")
        generator = YouTubeSyntheticDataGenerator(seed=42)
        users_df, videos_df, interactions_df, _ = generator.generate_all(
            num_users=10000, 
            num_videos=50000, 
            num_interactions=500000, 
            num_follows=50000
        )
    else:
        logger.info("Loading YouTube dataset from raw data directory.")
        users_df = pd.read_csv(user_path)
        videos_df = pd.read_csv(video_path)
        interactions_df = pd.read_csv(interaction_path)
        
    return users_df, videos_df, interactions_df

if __name__ == "__main__":
    # Test execution
    generator = YouTubeSyntheticDataGenerator(seed=42)
    generator.generate_all(num_users=100, num_videos=500, num_interactions=5000, num_follows=500)
