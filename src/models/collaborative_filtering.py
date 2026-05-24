import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from src.utils.logger import logger

class CollaborativeFilteringRecommender:
    """
    Memory-based Collaborative Filtering supporting both User-User similarity 
    and Item-Item similarity candidate retrieval.
    """
    def __init__(self, kind: str = "user", k: int = 10):
        self.kind = kind  # "user" or "item"
        self.k = k        # Number of neighbors to consider
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.video_to_idx = {}
        self.idx_to_video = {}

    def fit(self, interactions_df: pd.DataFrame):
        """
        Builds the sparse user-item interaction matrix and calculates similarity matrices.
        """
        logger.info(f"Fitting memory-based Collaborative Filtering ({self.kind}-based).")
        
        # Keep positive click items
        clicked_df = interactions_df[interactions_df["click"] == 1].copy()
        
        # Check if interactions exist
        if len(clicked_df) == 0:
            logger.warning("No clicked interactions to train Collaborative Filtering.")
            return
            
        # Re-assign indices
        unique_users = clicked_df["user_id"].unique()
        unique_videos = clicked_df["video_id"].unique()
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.video_to_idx = {vid: idx for idx, vid in enumerate(unique_videos)}
        self.idx_to_video = {idx: vid for vid, idx in self.video_to_idx.items()}
        
        num_users = len(unique_users)
        num_videos = len(unique_videos)
        
        rows = clicked_df["user_id"].map(self.user_to_idx).values
        cols = clicked_df["video_id"].map(self.video_to_idx).values
        
        # Build implicit rating values based on watch_ratio or just 1s
        vals = clicked_df["watch_ratio"].values
        # Add tiny eps to avoid zeros
        vals = np.maximum(vals, 0.1)
        
        # Construct CSR sparse matrix
        self.user_item_matrix = csr_matrix((vals, (rows, cols)), shape=(num_users, num_videos))
        
        # Compute Cosine Similarity
        if self.kind == "user":
            logger.info("Computing User-User similarity matrix.")
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
            np.fill_diagonal(self.similarity_matrix, 0) # Remove self similarity
        else:
            logger.info("Computing Item-Item similarity matrix.")
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            np.fill_diagonal(self.similarity_matrix, 0) # Remove self similarity

        logger.info(f"Collaborative Filtering fit complete. Matrix shape: {self.user_item_matrix.shape}")

    def retrieve_candidates(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieves recommended candidate video IDs for a given user.
        Returns:
            List of (video_id, similarity_score)
        """
        if user_id not in self.user_to_idx:
            logger.debug(f"User {user_id} not seen during training. Falling back to empty candidates.")
            return []
            
        user_idx = self.user_to_idx[user_id]
        
        if self.kind == "user":
            return self._retrieve_user_based(user_idx, top_n)
        else:
            return self._retrieve_item_based(user_idx, top_n)

    def _retrieve_user_based(self, user_idx: int, top_n: int) -> List[Tuple[int, float]]:
        # Get similarities for the target user
        user_sims = self.similarity_matrix[user_idx]
        
        # Find top k similar users
        similar_users = np.argsort(user_sims)[::-1][:self.k]
        
        # Aggregate their watch logs weighted by similarity
        scores = np.zeros(self.user_item_matrix.shape[1])
        for other_user in similar_users:
            sim_weight = user_sims[other_user]
            if sim_weight <= 0:
                continue
            # Weighted ratings
            scores += sim_weight * self.user_item_matrix[other_user].toarray().flatten()
            
        # Zero out items the target user has already interacted with
        interacted_items = self.user_item_matrix[user_idx].toarray().flatten() > 0
        scores[interacted_items] = 0
        
        # Sort and return video IDs
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(self.idx_to_video[idx]), float(scores[idx])))
                
        return results

    def _retrieve_item_based(self, user_idx: int, top_n: int) -> List[Tuple[int, float]]:
        # Get items user has already clicked
        user_profile = self.user_item_matrix[user_idx].toarray().flatten()
        interacted_indices = np.where(user_profile > 0)[0]
        
        if len(interacted_indices) == 0:
            return []
            
        # Average similarity of all items in database to items user watched
        # Shape: [Num_Videos]
        scores = np.zeros(self.user_item_matrix.shape[1])
        for item_idx in interacted_indices:
            weight = user_profile[item_idx]
            scores += weight * self.similarity_matrix[item_idx]
            
        # Zero out items user has already interacted with
        scores[interacted_indices] = 0
        
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(self.idx_to_video[idx]), float(scores[idx])))
                
        return results
