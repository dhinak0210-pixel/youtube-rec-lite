import pandas as pd
import numpy as np
from typing import List, Tuple
from src.utils.logger import logger

class ColdStartHandler:
    """
    Handles recommendation routing when a user or video is "cold" (lacks historical logs).
    Implements a category-level trending fallback index to serve highly engaging 
    videos within a user's specified niche preferences.
    """
    def __init__(self, users_df: pd.DataFrame, videos_df: pd.DataFrame):
        self.users_df = users_df
        self.videos_df = videos_df
        self.global_popular_videos = []
        self.category_popular_videos = {}
        
    def fit(self, interactions_df: pd.DataFrame):
        """
        Pre-calculates trending popularity charts globally and independently for all 20 categories.
        """
        logger.info("Fitting Category-based Cold Start index maps.")
        
        clicked_df = interactions_df[interactions_df["click"] == 1].copy()
        
        if len(clicked_df) == 0:
            logger.warning("Zero interaction logs for Cold Start fitting. Falling back to metadata ranking.")
            self.global_popular_videos = (
                self.videos_df.sort_values(by="popularity_weight", ascending=False)["video_id"]
                .head(30).tolist()
            )
            
            # Map category outlines
            for cat in self.videos_df["category"].unique():
                cat_vids = self.videos_df[self.videos_df["category"] == cat]
                self.category_popular_videos[cat] = (
                    cat_vids.sort_values(by="popularity_weight", ascending=False)["video_id"]
                    .head(20).tolist()
                )
            return
            
        # 1. Global trending popularities
        global_pop = clicked_df.groupby("video_id").size().reset_index(name="clicks")
        global_pop = global_pop.sort_values(by="clicks", ascending=False)
        self.global_popular_videos = global_pop["video_id"].head(40).tolist()
        
        # 2. Category-specific popularities (Trending by Niche)
        # Merge interaction clicks with video category metadata
        merged = clicked_df.merge(self.videos_df[["video_id", "category"]], on="video_id", how="left")
        
        for cat in self.videos_df["category"].unique():
            cat_clicks = merged[merged["category"] == cat]
            if len(cat_clicks) > 0:
                cat_pop = cat_clicks.groupby("video_id").size().reset_index(name="clicks")
                cat_pop = cat_pop.sort_values(by="clicks", ascending=False)
                self.category_popular_videos[cat] = cat_pop["video_id"].head(20).tolist()
            else:
                # If no clicks, fallback to popularity weights
                cat_vids = self.videos_df[self.videos_df["category"] == cat]
                self.category_popular_videos[cat] = (
                    cat_vids.sort_values(by="popularity_weight", ascending=False)["video_id"]
                    .head(20).tolist()
                )
                
        logger.info("Category-based Cold Start pre-indexing complete.")

    def get_user_fallback(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieves personalized cold start candidates based on listed preferred categories.
        """
        user_rows = self.users_df[self.users_df["user_id"] == user_id]
        
        if len(user_rows) == 0:
            logger.debug(f"Cold Start: User {user_id} completely anonymous. Serving global charts.")
            popular_list = self.global_popular_videos
        else:
            pref_cats_str = user_rows.iloc[0]["preferred_categories"]
            pref_cats = pref_cats_str.split("|")
            logger.debug(f"Cold Start: User {user_id} prefers {pref_cats}. Blending trending videos.")
            
            # Blend popular videos from their preferred categories
            blended_candidates = []
            for cat in pref_cats:
                cat_recs = self.category_popular_videos.get(cat, [])
                blended_candidates.extend(cat_recs)
                
            # Deduplicate while preserving order across niches
            seen = set()
            popular_list = []
            for vid in blended_candidates:
                if vid not in seen:
                    seen.add(vid)
                    popular_list.append(vid)
                    
            # Pad with global popular if pool is too small
            if len(popular_list) < top_n:
                for vid in self.global_popular_videos:
                    if vid not in seen:
                        seen.add(vid)
                        popular_list.append(vid)
                        
        return [
            (int(vid), float(1.0 - (idx / len(popular_list))))
            for idx, vid in enumerate(popular_list[:top_n])
        ]

    def get_item_fallback(self, video_id: int, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Finds content-similar items for a cold video based on category sharing.
        """
        video_rows = self.videos_df[self.videos_df["video_id"] == video_id]
        if len(video_rows) == 0:
            return [(int(vid), 1.0) for vid in self.global_popular_videos[:top_n]]
            
        category = video_rows.iloc[0]["category"]
        similar_videos = self.videos_df[
            (self.videos_df["category"] == category) & 
            (self.videos_df["video_id"] != video_id)
        ]
        
        # Sort by inherent popularity weight
        similar_videos = similar_videos.sort_values(by="popularity_weight", ascending=False)
        results = similar_videos["video_id"].head(top_n).tolist()
        
        return [
            (int(vid), float(1.0 - (idx / len(results))))
            for idx, vid in enumerate(results)
        ]
