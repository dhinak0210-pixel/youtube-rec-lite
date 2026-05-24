"""
Memory-based Item-Item Collaborative Filtering model for YouTube recommendations.

This module computes similarities between video catalog items by representing them as 
vectors of user interaction weights, applying IDF downweighting for highly popular/generic videos,
and computing unit-normalized cosine similarity scores to yield recommendation candidates.
"""

import os
from collections import defaultdict
import math
from typing import List as TypeList, Set as TypeSet
from loguru import logger
import numpy as np
import scipy.sparse
from scipy.sparse import diags, csr_matrix
from config import settings
from data.schemas import Interaction, ScoredItem

class CollaborativeFilter:
    """Item-Based Collaborative Filtering engine with sparse matrix computations and IDF scaling."""

    def __init__(self, num_users: int, num_items: int, num_neighbors: int):
        """Initializes state parameters, similarity maps, and user histories."""
        self.num_users = num_users
        self.num_items = num_items
        self.num_neighbors = num_neighbors
        
        # Maps item_id -> list of (similar_item_id, score)
        self.item_similarities = {}
        
        # Maps user_id -> set of item_ids they interacted with
        self.user_history = defaultdict(set)
        
        self._fitted = False

    def fit(self, interactions: TypeList[Interaction]) -> 'CollaborativeFilter':
        """Fits item similarity maps by constructing a sparse user-item interaction matrix."""
        logger.info("Fitting CollaborativeFilter model...")
        
        if not interactions:
            logger.warning("Empty interactions list passed. Setting fitted status to True with empty states.")
            self._fitted = True
            return self

        # 1. Map interactions with positive weights to row/col indices
        row_indices = []
        col_indices = []
        data_values = []

        for inter in interactions:
            if inter.weight > 0:
                # Boundary check coordinates
                u_id = inter.user_id
                i_id = inter.item_id
                
                if 0 <= u_id < self.num_users and 0 <= i_id < self.num_items:
                    row_indices.append(u_id)
                    col_indices.append(i_id)
                    data_values.append(inter.weight)
                    self.user_history[u_id].add(i_id)

        if not row_indices:
            logger.warning("No interactions satisfied the positive weight constraints.")
            self._fitted = True
            return self

        # 2. Build sparse CSR matrix of shape (num_users, num_items)
        matrix = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(self.num_users, self.num_items),
            dtype=float
        )

        # 3. Apply Inverse Document Frequency (IDF) weighting to downweight popular items
        item_interaction_counts = np.bincount(col_indices, minlength=self.num_items)
        idf = np.log(self.num_users / (item_interaction_counts + 1.0))

        # Multiply columns by their respective IDF vector weights using a diagonal matrix
        idf_diag = diags(idf)
        weighted_matrix = matrix.dot(idf_diag)

        # 4. Compute cosine similarity between item columns
        self._compute_similarities(weighted_matrix, col_indices=col_indices)
        
        self._fitted = True
        logger.info(f"CollaborativeFilter fitted successfully! Items with similarities: {len(self.item_similarities)}")
        return self

    def _compute_similarities(self, matrix: csr_matrix, col_indices: TypeList[int], max_items: int = 2000):
        """Finds most active items, extracts their vectors, normalizes them, and stores similarities."""
        item_counts = np.bincount(col_indices, minlength=self.num_items)
        popular_item_indices = np.argsort(item_counts)[::-1]
        
        # Retain only items with at least one active interaction
        active_items = [int(idx) for idx in popular_item_indices if item_counts[idx] > 0]
        candidate_items = active_items[:max_items]

        if not candidate_items:
            return

        # Convert to CSC format for fast column slice extractions
        csc_matrix = matrix.tocsc()
        submatrix = csc_matrix[:, candidate_items]

        # Compute column L2 norms
        norms = np.sqrt(np.asarray(submatrix.power(2).sum(axis=0)).flatten())
        norms[norms == 0.0] = 1e-9  # Avoid division by zero

        # Unit-normalize the item vectors
        norm_diag = diags(1.0 / norms)
        normalized_submatrix = submatrix.dot(norm_diag)

        # Dot product of unit vectors computes cosine similarity
        similarity = normalized_submatrix.T.dot(normalized_submatrix).toarray()

        # For each item, store top neighbors excluding self-similarity
        num_candidates = len(candidate_items)
        for i in range(num_candidates):
            item_i = candidate_items[i]
            sim_scores = similarity[i]
            
            # Sort similarity indexes descending
            sorted_indices = np.argsort(sim_scores)[::-1]
            
            sim_list = []
            for idx in sorted_indices:
                item_j = candidate_items[idx]
                if item_i == item_j:
                    continue
                score = float(sim_scores[idx])
                if score <= 0.0:
                    continue
                sim_list.append((item_j, score))
                if len(sim_list) >= self.num_neighbors:
                    break
            
            if sim_list:
                self.item_similarities[item_i] = sim_list

    def predict(self, user_id: int, num_candidates: int = 100, exclude_items: TypeSet[int] = None) -> TypeList[ScoredItem]:
        """Predicts recommendation candidate list by aggregating historical similarity vectors."""
        if not self._fitted:
            logger.warning("Cannot call predict on an unfitted model.")
            return []

        user_watched = self.user_history.get(user_id, set())
        if not user_watched:
            return []

        scores = {}
        for item_id in user_watched:
            similar_items = self.item_similarities.get(item_id, [])
            for sim_item_id, score in similar_items:
                scores[sim_item_id] = scores.get(sim_item_id, 0.0) + score

        # Aggregate exclusions
        exclude = set(user_watched)
        if exclude_items:
            exclude.update(exclude_items)

        filtered_scores = {iid: score for iid, score in scores.items() if iid not in exclude}
        sorted_items = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:num_candidates]

        return [
            ScoredItem(item_id=iid, score=score, source="cf")
            for iid, score in sorted_items
        ]

    def get_history(self, user_id: int) -> TypeSet[int]:
        """Retrieves user interaction click history."""
        return self.user_history.get(user_id, set())


# ==========================================
# 🔄 Backward Compatibility Wrapper Class
# ==========================================

class CollaborativeFilteringRecommender:
    """Wrapper Collaborative Filtering Recommender to support offline legacy systems."""

    def __init__(self):
        self.num_neighbors = settings.cf_num_neighbors
        self.similarity_matrix = None
        self.user_history = {}
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

    def fit(self, users=None, items=None, interactions=None, data_dir: str = "data"):
        """Fits the item similarity matrix using Pydantic profile lists or loaded CSV datasets."""
        import pandas as pd
        from sklearn.metrics.pairwise import cosine_similarity
        
        if users is not None and items is not None and interactions is not None:
            # Pydantic v2 profile lists
            u_ids = sorted(list(set(u.user_id for u in users)))
            i_ids = sorted(list(set(i.item_id for i in items)))
            
            self.user_to_idx = {uid: idx for idx, uid in enumerate(u_ids)}
            self.item_to_idx = {iid: idx for idx, iid in enumerate(i_ids)}
            self.idx_to_item = {idx: iid for idx, iid in enumerate(i_ids)}
            
            interaction_matrix = np.zeros((len(u_ids), len(i_ids)))
            self.user_history = {}
            
            for inter in interactions:
                if inter.weight > 1.0:
                    self.user_history.setdefault(inter.user_id, []).append(inter.item_id)
                
                if inter.user_id in self.user_to_idx and inter.item_id in self.item_to_idx:
                    uidx = self.user_to_idx[inter.user_id]
                    iidx = self.item_to_idx[inter.item_id]
                    interaction_matrix[uidx, iidx] += inter.weight
                    
        else:
            # CSV loading
            interactions_path = os.path.join(data_dir, "interactions.csv")
            if not os.path.exists(interactions_path):
                raise FileNotFoundError(f"Missing interactions logs at: {interactions_path}")
                
            df = pd.read_csv(interactions_path)
            click_df = df[df["click"] == 1]
            
            pivot = click_df.pivot_table(
                index="user_id", 
                columns="video_id", 
                values="watch_ratio", 
                fill_value=0.0
            )
            
            u_ids = pivot.index.tolist()
            i_ids = pivot.columns.tolist()
            
            self.user_to_idx = {uid: idx for idx, uid in enumerate(u_ids)}
            self.item_to_idx = {vid: idx for idx, vid in enumerate(i_ids)}
            self.idx_to_item = {idx: vid for idx, vid in enumerate(i_ids)}
            
            interaction_matrix = pivot.values
            
            self.user_history = {}
            for uid, group in click_df.groupby("user_id"):
                self.user_history[uid] = group["video_id"].tolist()
                
        # Compute cosine similarity
        if interaction_matrix.shape[0] > 0 and interaction_matrix.shape[1] > 0:
            self.similarity_matrix = cosine_similarity(interaction_matrix.T)
        else:
            self.similarity_matrix = np.zeros((len(i_ids), len(i_ids)))
            
        print(f"✅ Collaborative Filtering fit complete. Mapped items: {len(self.item_to_idx)}")

    def recommend(self, user_id: int, top_n: int = 10) -> TypeList[tuple]:
        """Retrieves nearest-neighbor candidates for a given user."""
        if self.similarity_matrix is None or user_id not in self.user_history:
            return []
            
        history = self.user_history[user_id]
        if not history:
            return []
            
        scores = np.zeros(len(self.item_to_idx))
        
        for vid in history:
            if vid not in self.item_to_idx:
                continue
            v_idx = self.item_to_idx[vid]
            scores += self.similarity_matrix[v_idx]
            
        for vid in history:
            if vid in self.item_to_idx:
                scores[self.item_to_idx[vid]] = -1.0
                
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                recommendations.append((self.idx_to_item[idx], float(scores[idx])))
                
        return recommendations
