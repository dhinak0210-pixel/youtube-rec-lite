import numpy as np
import pandas as pd
from typing import List, Tuple
from src.config import EMBEDDING_DIM
from src.utils.logger import logger

class ALSMatrixFactorization:
    """
    Alternating Least Squares (ALS) Matrix Factorization for implicit interaction signals.
    Factorizes the user-item interaction matrix into latent representation matrices:
    Preference = User_Factors (U) . Item_Factors (V)^T
    """
    def __init__(self, latent_dims: int = EMBEDDING_DIM, lambda_reg: float = 0.1, epochs: int = 15):
        self.latent_dims = latent_dims
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.user_factors = None
        self.item_factors = None
        
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.video_to_idx = {}
        self.idx_to_video = {}

    def fit(self, interactions_df: pd.DataFrame):
        """
        Fits user and item latent matrices using the iterative ALS protocol.
        """
        logger.info(f"Fitting ALS Matrix Factorization (Latent Dims: {self.latent_dims}, Epochs: {self.epochs}).")
        
        # Keep positive click items
        clicked_df = interactions_df[interactions_df["click"] == 1].copy()
        if len(clicked_df) == 0:
            logger.warning("No interactions to train ALS.")
            return

        unique_users = clicked_df["user_id"].unique()
        unique_videos = clicked_df["video_id"].unique()
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.video_to_idx = {vid: idx for idx, vid in enumerate(unique_videos)}
        self.idx_to_video = {idx: vid for vid, idx in self.video_to_idx.items()}
        
        num_users = len(unique_users)
        num_videos = len(unique_videos)
        
        # 1. Build interaction confidence matrix C and preferences P
        # C = 1 + alpha * rating, we set alpha = 40 (standard Hu, Koren, Volinsky parameter)
        C = np.zeros((num_users, num_videos))
        P = np.zeros((num_users, num_videos))
        
        for _, row in clicked_df.iterrows():
            u_idx = self.user_to_idx[row["user_id"]]
            v_idx = self.video_to_idx[row["video_id"]]
            watch_ratio = row["watch_ratio"]
            
            P[u_idx, v_idx] = 1.0
            C[u_idx, v_idx] = 1.0 + 40.0 * watch_ratio

        # Initialize Latent Factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(scale=1.0/self.latent_dims, size=(num_users, self.latent_dims))
        self.item_factors = np.random.normal(scale=1.0/self.latent_dims, size=(num_videos, self.latent_dims))
        
        # ALS optimization loop
        # Alternates solving for Users (holding Items constant) and Items (holding Users constant)
        for epoch in range(self.epochs):
            # Solve for Users: U_i = (V^T * C^i * V + lambda * I)^(-1) * V^T * C^i * P_i
            VTV = self.item_factors.T @ self.item_factors
            reg_I = self.lambda_reg * np.eye(self.latent_dims)
            
            for i in range(num_users):
                Ci = np.diag(C[i, :] - 1.0) # Diagonal difference matrix (C^i - I)
                # Compute V^T * C^i * V = V^T * V + V^T * (C^i - I) * V
                # We optimize this step:
                VTCiV = VTV + self.item_factors.T @ Ci @ self.item_factors
                A = VTCiV + reg_I
                b = self.item_factors.T @ (C[i, :] * P[i, :])
                self.user_factors[i] = np.linalg.solve(A, b)
                
            # Solve for Items: V_j = (U^T * C^j * U + lambda * I)^(-1) * U^T * C^j * P_j
            UTU = self.user_factors.T @ self.user_factors
            
            for j in range(num_videos):
                Cj = np.diag(C[:, j] - 1.0)
                UTUj = UTU + self.user_factors.T @ Cj @ self.user_factors
                A = UTUj + reg_I
                b = self.user_factors.T @ (C[:, j] * P[:, j])
                self.item_factors[j] = np.linalg.solve(A, b)
                
            # Compute loss for debugging/logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                pred = self.user_factors @ self.item_factors.T
                loss = np.sum(C * (P - pred) ** 2) + self.lambda_reg * (np.sum(self.user_factors**2) + np.sum(self.item_factors**2))
                logger.info(f"ALS Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}")
                
        logger.info("ALS model training finished.")

    def retrieve_candidates(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieves top candidate video IDs by computing latent factor dot products.
        """
        if user_id not in self.user_to_idx:
            logger.debug(f"User {user_id} not seen in ALS training. Returning empty.")
            return []
            
        u_idx = self.user_to_idx[user_id]
        user_vector = self.user_factors[u_idx]
        
        # Calculate scores for all items
        scores = self.item_factors @ user_vector
        
        # Sort scores in descending order
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        results = [
            (int(self.idx_to_video[idx]), float(scores[idx]))
            for idx in top_indices
        ]
        return results
