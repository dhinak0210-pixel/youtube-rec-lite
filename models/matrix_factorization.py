"""
Implicit Alternating Least Squares (ALS) Matrix Factorization recommender.

Models binary preferences and positive watch-time confidence intervals using
regularized linear solves. Captures latent user/item semantic embeddings
and retrieves implicit candidates.
"""

import os
from collections import defaultdict
from typing import List as TypeList, Set as TypeSet, Optional
from loguru import logger
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
from config import settings
from data.schemas import Interaction, ScoredItem

class MatrixFactorization:
    """Alternating Least Squares (ALS) Matrix Factorization engine for implicit feedback datasets."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 32,
        regularization: float = 0.02,
        iterations: int = 10,
        alpha: float = 40.0
    ):
        """Initializes latent dimensions, regularization rates, and factor arrays."""
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        
        self.user_factors = None
        self.item_factors = None
        self.user_history = defaultdict(set)
        self._fitted = False

    def fit(self, interactions: TypeList[Interaction]) -> 'MatrixFactorization':
        """Trains user and item latent matrices using implicit ALS updates."""
        logger.info("Fitting Implicit ALS MatrixFactorization recommender...")
        
        # 1. Build preference (R) and confidence (C) matrices
        R, C = self._build_matrices(interactions)
        
        # 2. Initialize latent factor weights
        rng = np.random.RandomState(42)
        self.user_factors = rng.normal(0, 0.01, (self.num_users, self.num_factors))
        self.item_factors = rng.normal(0, 0.01, (self.num_items, self.num_factors))
        
        # Pre-build identity regularization matrix
        reg_I = self.regularization * np.identity(self.num_factors)
        
        # 3. Alternate training steps
        for i in range(self.iterations):
            # Update user factors (axis=0)
            self._als_step(self.user_factors, self.item_factors, R, C, reg_I)
            
            # Update item factors (axis=1) using transposed preference and confidence CSR representations
            R_T = R.T.tocsr()
            C_T = C.T.tocsr()
            self._als_step(self.item_factors, self.user_factors, R_T, C_T, reg_I)
            
            # Periodically evaluate sample loss
            if (i + 1) % 3 == 0 or (i + 1) == self.iterations:
                loss = self._sample_loss(R, C, n=500)
                logger.info(f"   ALS Iteration {i+1}/{self.iterations} - Weighted Sample Loss: {loss:.5f}")
                
        self._fitted = True
        logger.info("Implicit ALS MatrixFactorization model fitted successfully!")
        return self

    def _build_matrices(self, interactions: TypeList[Interaction]) -> tuple[csr_matrix, csr_matrix]:
        """Assembles binary preference (R) and watch-time confidence (C) CSR matrices."""
        row_indices = []
        col_indices = []
        r_data = []
        c_data = []
        
        self.user_history = defaultdict(set)

        for inter in interactions:
            if inter.weight > 0:
                u_id = inter.user_id
                i_id = inter.item_id
                
                # Check valid dimensions
                if 0 <= u_id < self.num_users and 0 <= i_id < self.num_items:
                    row_indices.append(u_id)
                    col_indices.append(i_id)
                    r_data.append(1.0)
                    c_data.append(1.0 + self.alpha * inter.weight)
                    self.user_history[u_id].add(i_id)

        R = csr_matrix((r_data, (row_indices, col_indices)), shape=(self.num_users, self.num_items), dtype=float)
        C = csr_matrix((c_data, (row_indices, col_indices)), shape=(self.num_users, self.num_items), dtype=float)
        
        return R, C

    def _als_step(
        self,
        target_factors: np.ndarray,
        fixed_factors: np.ndarray,
        R: csr_matrix,
        C: csr_matrix,
        reg_I: np.ndarray
    ):
        """Executes a single regularized least squares projection step across target factors."""
        # Precompute static covariance dot product for extreme speed boost
        fixed_T_fixed = fixed_factors.T @ fixed_factors
        
        for i in range(target_factors.shape[0]):
            row_start, row_end = C.indptr[i], C.indptr[i+1]
            if row_start == row_end:
                # Handle edge case: empty user or item history
                target_factors[i] = np.zeros(self.num_factors)
                continue
                
            indices = C.indices[row_start:row_end]
            confidence_values = C.data[row_start:row_end]
            preference_values = R.data[row_start:row_end]
            
            V_i = fixed_factors[indices]
            
            # CuV = V_i * confidence_values (column-wise multiplication)
            CuV = V_i * confidence_values[:, np.newaxis]
            
            # Compute covariance coefficients: Y^T C^i Y + lambda I
            # Hu & Koren: Y^T C^i Y = Y^T Y + Y^T (C^i - I) Y
            A = fixed_T_fixed + V_i.T @ (CuV - V_i) + reg_I
            
            # Compute projection target vector: Y^T C^i p_i
            b = CuV.T @ preference_values
            
            # Solve regularized system of linear equations
            target_factors[i] = np.linalg.solve(A, b)

    def _sample_loss(self, R: csr_matrix, C: csr_matrix, n: int = 500) -> float:
        """Computes exact Hu-Koren implicit loss over sampled user rows."""
        n = min(n, self.num_users)
        if n <= 0:
            return 0.0
            
        rng = np.random.RandomState(99)
        sampled_users = rng.choice(self.num_users, size=n, replace=False)
        
        total_error = 0.0
        for u in sampled_users:
            row_start, row_end = C.indptr[u], C.indptr[u+1]
            interacted_items = C.indices[row_start:row_end]
            conf_vals = C.data[row_start:row_end]
            pref_vals = R.data[row_start:row_end]
            
            # Predict scores for all catalog items
            pred_all = self.user_factors[u] @ self.item_factors.T
            
            # Error for interacted items: confidence * (preference - prediction)^2
            error_interacted = np.sum(conf_vals * (pref_vals - pred_all[interacted_items])**2)
            
            # Error for non-interacted items: 1.0 * (0.0 - prediction)^2
            error_non_interacted = np.sum(pred_all**2) - np.sum(pred_all[interacted_items]**2)
            
            total_error += error_interacted + error_non_interacted
            
        return total_error / n

    def predict(
        self,
        user_id: int,
        num_candidates: int = 100,
        exclude_items: TypeSet[int] = None
    ) -> TypeList[ScoredItem]:
        """Predicts recommendation candidate list by computing latent dot products."""
        if not self._fitted or user_id >= self.num_users or user_id < 0:
            return []

        # Predict scores for all catalog items
        scores = self.user_factors[user_id] @ self.item_factors.T
        
        # Compile exclusions
        exclude = set(self.user_history.get(user_id, set()))
        if exclude_items:
            exclude.update(exclude_items)
            
        for item_id in exclude:
            if 0 <= item_id < self.num_items:
                scores[item_id] = -np.inf
                
        # Retrieve top indices using high-performance partitions (argpartition)
        if num_candidates >= self.num_items:
            top_indices = np.argsort(scores)[::-1]
        else:
            partitioned = np.argpartition(scores, -num_candidates)
            top_candidates_idx = partitioned[-num_candidates:]
            top_indices = top_candidates_idx[np.argsort(scores[top_candidates_idx])[::-1]]
            
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if np.isneginf(score):
                continue
            results.append(ScoredItem(item_id=int(idx), score=score, source="mf"))
            
        return results

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Retrieves user embedding array."""
        if self._fitted and 0 <= user_id < self.num_users:
            return self.user_factors[user_id].copy()
        return None

    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """Retrieves item embedding array."""
        if self._fitted and 0 <= item_id < self.num_items:
            return self.item_factors[item_id].copy()
        return None


# ==========================================
# 🔄 Backward Compatibility Legacy Class
# ==========================================

class MatrixFactorizationNet(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_factors: int):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)
        dot = (u_emb * i_emb).sum(dim=1, keepdim=True)
        u_b = self.user_bias(users)
        i_b = self.item_bias(items)
        prediction = dot + u_b + i_b + self.global_bias
        return prediction.squeeze()

class PyTorchMatrixFactorization:
    def __init__(self):
        self.num_factors = settings.mf_num_factors
        self.epochs = settings.mf_epochs
        self.model = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_history = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, users=None, items=None, interactions=None, data_dir: str = "data"):
        import pandas as pd
        
        if users is not None and items is not None and interactions is not None:
            unique_users = sorted(list(set(u.user_id for u in users)))
            unique_items = sorted(list(set(i.item_id for i in items)))
            
            self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
            self.item_to_idx = {vid: idx for idx, vid in enumerate(unique_items)}
            self.idx_to_item = {idx: vid for idx, vid in enumerate(unique_items)}
            
            self.user_history = {}
            for inter in interactions:
                if inter.weight > 1.0:
                    self.user_history.setdefault(inter.user_id, []).append(inter.item_id)
            
            user_indices = []
            item_indices = []
            watch_ratios = []
            for inter in interactions:
                if inter.user_id in self.user_to_idx and inter.item_id in self.item_to_idx:
                    user_indices.append(self.user_to_idx[inter.user_id])
                    item_indices.append(self.item_to_idx[inter.item_id])
                    watch_ratios.append(inter.watch_percentage)
                    
            user_tensor = torch.tensor(user_indices, dtype=torch.long)
            item_tensor = torch.tensor(item_indices, dtype=torch.long)
            y_tensor = torch.tensor(watch_ratios, dtype=torch.float32)
            
        else:
            interactions_path = os.path.join(data_dir, "interactions.csv")
            if not os.path.exists(interactions_path):
                raise FileNotFoundError(f"Missing interactions at: {interactions_path}")
                
            df = pd.read_csv(interactions_path)
            unique_users = df["user_id"].unique()
            unique_items = df["video_id"].unique()
            
            self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
            self.item_to_idx = {vid: idx for idx, vid in enumerate(unique_items)}
            self.idx_to_item = {idx: vid for idx, vid in enumerate(unique_items)}
            
            for uid, group in df[df["click"] == 1].groupby("user_id"):
                self.user_history[uid] = group["video_id"].tolist()
                
            user_tensor = torch.tensor([self.user_to_idx[uid] for uid in df["user_id"]], dtype=torch.long)
            item_tensor = torch.tensor([self.item_to_idx[vid] for vid in df["video_id"]], dtype=torch.long)
            y_tensor = torch.tensor(df["watch_ratio"].values, dtype=torch.float32)
            
        if len(user_tensor) == 0:
            return

        self.model = MatrixFactorizationNet(
            num_users=len(self.user_to_idx),
            num_items=len(self.item_to_idx),
            num_factors=self.num_factors
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-5)
        
        dataset = torch.utils.data.TensorDataset(user_tensor, item_tensor, y_tensor)
        batch_size = min(2048, max(1, len(dataset)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_users, batch_items, targets in loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_users, batch_items)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def recommend(self, user_id: int, top_n: int = 10) -> TypeList[tuple]:
        if self.model is None or user_id not in self.user_to_idx:
            return []
            
        self.model.eval()
        u_idx = self.user_to_idx[user_id]
        history = self.user_history.get(user_id, [])
        
        with torch.no_grad():
            user_tensor = torch.tensor([u_idx], dtype=torch.long).to(self.device)
            item_indices = torch.tensor(list(self.item_to_idx.values()), dtype=torch.long).to(self.device)
            
            user_embed = self.model.user_embedding(user_tensor).repeat(len(item_indices), 1)
            item_embed = self.model.item_embedding(item_indices)
            user_bias = self.model.user_bias(user_tensor).repeat(len(item_indices), 1)
            item_bias = self.model.item_bias(item_indices)
            
            scores = (user_embed * item_embed).sum(dim=1, keepdim=True) + user_bias + item_bias + self.model.global_bias
            scores = scores.squeeze()
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            scores = scores.cpu().numpy()
            
        for vid in history:
            if vid in self.item_to_idx:
                scores[self.item_to_idx[vid]] = -1e5
                
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = []
        for idx in top_indices:
            vid = self.idx_to_item[idx]
            recommendations.append((vid, float(scores[idx])))
            
        return recommendations

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Retrieves user embedding array from PyTorch model."""
        if self.model is None or user_id not in self.user_to_idx:
            return None
        self.model.eval()
        u_idx = self.user_to_idx[user_id]
        with torch.no_grad():
            u_tensor = torch.tensor([u_idx], dtype=torch.long).to(self.device)
            emb = self.model.user_embedding(u_tensor).squeeze().cpu().numpy()
        return emb

    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """Retrieves item embedding array from PyTorch model."""
        if self.model is None or item_id not in self.item_to_idx:
            return None
        self.model.eval()
        i_idx = self.item_to_idx[item_id]
        with torch.no_grad():
            i_tensor = torch.tensor([i_idx], dtype=torch.long).to(self.device)
            emb = self.model.item_embedding(i_tensor).squeeze().cpu().numpy()
        return emb

