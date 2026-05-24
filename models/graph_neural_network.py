"""
Bipartite and Social-aware Graph Neural Network (GNN) models in PyTorch.

Contains PyTorch convolutional SAGE layers and social relation engines.
Integrates user demographics, catalog interactions, and friend connections
to classify link engagement probabilities.
"""

import os
from collections import defaultdict
from typing import List as TypeList, Dict as TypeDict, Set as TypeSet, Optional, Tuple as TypeTuple
from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import settings
from data.schemas import Interaction, ScoredItem

# ==========================================
# 🌐 Part 1 - Social Graph Context
# ==========================================

class SocialGraph:
    """Stores directional follower relationships and user item viewing history indices."""

    def __init__(self):
        """Initializes mapping indices for user networks and video views."""
        self.followees = defaultdict(list)    # follower -> list of followees
        self.item_viewers = defaultdict(list)  # item_id -> list of users who watched it

    def add_follow(self, follower: int, followee: int):
        """Appends follower link connection."""
        self.followees[follower].append(followee)

    def add_view(self, user_id: int, item_id: int):
        """Appends item viewing record."""
        self.item_viewers[item_id].append(user_id)

    def get_friends(self, user_id: int, max_k: int = 20) -> TypeList[int]:
        """Retrieves list of active user connections up to max_k."""
        friends = self.followees.get(user_id, [])
        if len(friends) > max_k:
            # Deterministically or randomly sample to satisfy length constraints
            rng = np.random.RandomState(42)
            return list(rng.choice(friends, size=max_k, replace=False))
        return friends

    def friend_watch_rate(self, user_id: int, item_id: int) -> float:
        """Calculates what ratio of user followees interacted with the given catalog item."""
        friends = set(self.followees.get(user_id, []))
        if not friends:
            return 0.0
        viewers = set(self.item_viewers.get(item_id, []))
        intersection = friends.intersection(viewers)
        return len(intersection) / max(len(friends), 1)

    def get_social_features(self, user_id: int, item_id: int) -> np.ndarray:
        """Extracts normalized real-valued social interaction vector of shape (4,)."""
        friends = self.followees.get(user_id, [])
        num_friends = len(friends)
        
        if num_friends == 0:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
        viewers = set(self.item_viewers.get(item_id, []))
        friends_watched = [f for f in friends if f in viewers]
        num_watched = len(friends_watched)
        
        feat_1 = num_watched / num_friends
        feat_2 = min(num_watched / 5.0, 1.0)
        feat_3 = 1.0 if num_watched > 0 else 0.0
        feat_4 = min(num_friends / 50.0, 1.0)
        
        return np.array([feat_1, feat_2, feat_3, feat_4], dtype=np.float32)


# ==========================================
# 🎛️ Part 2 - GraphSAGE Conv Layer
# ==========================================

class SAGEConv(nn.Module):
    """GraphSAGE convolutional message aggregating layer using average pool representations."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, self_feat: torch.Tensor, neighbor_feat: torch.Tensor) -> torch.Tensor:
        # self_feat shape: (B, in_dim)
        # neighbor_feat shape: (B, K, in_dim)
        agg = neighbor_feat.mean(dim=1)  # shape: (B, in_dim)
        h = torch.cat([self_feat, agg], dim=-1)  # shape: (B, in_dim * 2)
        return self.norm(self.relu(self.linear(h)))


# ==========================================
# 🌲 Part 3 - GNN Recommender Net
# ==========================================

class GNNRecommender(nn.Module):
    """Link predictor scoring user-item engagement margins using SAGE and social representations."""

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)  # Declared for future deep layers
        
        # Scorer projection network
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + emb_dim + 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        neighbor_ids: torch.Tensor,
        social_feats: torch.Tensor
    ) -> torch.Tensor:
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        n = self.user_emb(neighbor_ids)  # Shape (B, K, emb_dim)
        
        h = self.conv1(u, n)  # Shape (B, hidden_dim)
        combined = torch.cat([h, v, social_feats], dim=-1)
        
        # Output classification score (0.0 to 1.0)
        return self.scorer(combined).squeeze(-1)


# ==========================================
# ⚡ Part 4 - SAGE Model Trainer
# ==========================================

class GNNTrainer:
    """Coordinates neighborhood assembly and trains GraphSAGE networks using BCELoss."""

    def __init__(self, num_users: int, num_items: int, max_neighbors: int = 10):
        self.num_users = num_users
        self.num_items = num_items
        self.max_neighbors = max_neighbors
        
        self.model = GNNRecommender(num_users, num_items)
        self.graph = SocialGraph()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._fitted = False

    def build_graph(self, interactions: TypeList[Interaction], social_graph: TypeDict[int, TypeList[int]]):
        """Assembles connections mapping within the social graph indices."""
        for inter in interactions:
            if inter.weight > 0:
                self.graph.add_view(inter.user_id, inter.item_id)
                
        for follower, followees in social_graph.items():
            for f in followees:
                self.graph.add_follow(follower, f)
                
        logger.info("Social graph loaded successfully!")

    def train(self, interactions: TypeList[Interaction], epochs: int = 5) -> TypeList[float]:
        """Fits user/item graph weights using balanced classification updates."""
        logger.info("Training social-aware GraphSAGE recommender...")
        
        # Build positive training pairs (user, item)
        pos_pairs = [
            (inter.user_id, inter.item_id)
            for inter in interactions
            if inter.weight > 1.0 and 0 <= inter.user_id < self.num_users and 0 <= inter.item_id < self.num_items
        ]
        
        if not pos_pairs:
            logger.warning("No valid positive user-item interaction pairs for GNN training. Skipping fit.")
            self._fitted = True
            return []
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        criterion = nn.BCELoss()
        
        losses = []
        batch_size = 256
        self.model.train()
        
        for epoch in range(epochs):
            # Shuffle positive samples
            rng = np.random.RandomState(42 + epoch)
            shuffled_indices = rng.permutation(len(pos_pairs))
            
            epoch_loss = 0.0
            num_batches = 0
            
            for start_idx in range(0, len(pos_pairs), batch_size):
                batch_indices = shuffled_indices[start_idx : start_idx + batch_size]
                batch_pos = [pos_pairs[idx] for idx in batch_indices]
                
                # Sample negative items to balance labels
                batch_u = []
                batch_i = []
                batch_y = []
                
                for u, i in batch_pos:
                    # Pos pair
                    batch_u.append(u)
                    batch_i.append(i)
                    batch_y.append(1.0)
                    
                    # Neg pair
                    neg_item = int(rng.randint(0, self.num_items))
                    batch_u.append(u)
                    batch_i.append(neg_item)
                    batch_y.append(0.0)
                    
                # Assemble neighbor ID tensors
                neighbors_array = self._sample_neighbors(batch_u)
                
                # Compile social feature arrays
                social_array = np.array([
                    self.graph.get_social_features(uid, iid)
                    for uid, iid in zip(batch_u, batch_i)
                ], dtype=np.float32)
                
                # Move training variables to hardware target
                u_tensor = torch.tensor(batch_u, dtype=torch.long).to(self.device)
                i_tensor = torch.tensor(batch_i, dtype=torch.long).to(self.device)
                n_tensor = torch.tensor(neighbors_array, dtype=torch.long).to(self.device)
                s_tensor = torch.tensor(social_array, dtype=torch.float32).to(self.device)
                y_tensor = torch.tensor(batch_y, dtype=torch.float32).to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(u_tensor, i_tensor, n_tensor, s_tensor)
                
                loss = criterion(pred, y_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / max(1, num_batches)
            losses.append(avg_loss)
            logger.info(f"   GraphSAGE Epoch {epoch+1}/{epochs} - Social BCELoss: {avg_loss:.5f}")
            
        self._fitted = True
        logger.info("GraphSAGE social recommender trained successfully!")
        return losses

    def _sample_neighbors(self, user_ids: TypeList[int]) -> np.ndarray:
        """Draws consistent neighborhoods of friends for each batch user."""
        neighbors_matrix = []
        
        for uid in user_ids:
            friends = self.graph.get_friends(uid, max_k=self.max_neighbors)
            # Pad with user itself if empty or insufficient
            if not friends:
                padded = [uid] * self.max_neighbors
            elif len(friends) < self.max_neighbors:
                padded = friends + [uid] * (self.max_neighbors - len(friends))
            else:
                padded = friends[:self.max_neighbors]
                
            # Keep within user embedding boundaries
            padded_clipped = [np.clip(f, 0, self.num_users - 1) for f in padded]
            neighbors_matrix.append(padded_clipped)
            
        return np.array(neighbors_matrix, dtype=np.int64)

    def score_items(self, user_id: int, item_ids: TypeList[int]) -> TypeList[TypeTuple[int, float]]:
        """Scores catalog items for user recommendation matches."""
        if not self._fitted or not item_ids:
            return [(iid, 0.5) for iid in item_ids]
            
        self.model.eval()
        
        # Replicate user across items
        batch_u = [user_id] * len(item_ids)
        batch_i = list(item_ids)
        
        neighbors_array = self._sample_neighbors(batch_u)
        social_array = np.array([
            self.graph.get_social_features(user_id, iid)
            for iid in item_ids
        ], dtype=np.float32)
        
        with torch.no_grad():
            u_tensor = torch.tensor(batch_u, dtype=torch.long).to(self.device)
            i_tensor = torch.tensor(batch_i, dtype=torch.long).to(self.device)
            n_tensor = torch.tensor(neighbors_array, dtype=torch.long).to(self.device)
            s_tensor = torch.tensor(social_array, dtype=torch.float32).to(self.device)
            
            scores = self.model(u_tensor, i_tensor, n_tensor, s_tensor).cpu().numpy()
            
        scored_pairs = [(iid, float(score)) for iid, score in zip(item_ids, scores)]
        # Sort descending
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs


# ==========================================
# 🔄 Backward Compatibility Legacy Class
# ==========================================

class BipartiteGNNNet(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, adj_matrix: torch.Tensor) -> tuple:
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        i_msg = torch.matmul(adj_matrix.t(), u_emb)
        i_deg = adj_matrix.sum(dim=0, keepdim=True).t() + 1e-9
        i_emb_updated = (i_emb + i_msg / i_deg) * 0.5
        
        u_msg = torch.matmul(adj_matrix, i_emb_updated)
        u_deg = adj_matrix.sum(dim=1, keepdim=True) + 1e-9
        u_emb_updated = (u_emb + u_msg / u_deg) * 0.5
        
        return u_emb_updated, i_emb_updated

class PyTorchGNNRecommender:
    def __init__(self):
        self.embedding_dim = settings.gnn_embedding_dim
        self.epochs = settings.gnn_epochs
        self.model = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.adj_matrix = None
        self.user_history = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, users=None, items=None, interactions=None, data_dir: str = "data"):
        if users is not None and items is not None and interactions is not None:
            unique_users = sorted(list(set(u.user_id for u in users)))
            unique_items = sorted(list(set(i.item_id for i in items)))
            
            self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
            self.item_to_idx = {vid: idx for idx, vid in enumerate(unique_items)}
            self.idx_to_item = {idx: vid for vid, idx in self.item_to_idx.items()}
            
            self.user_history = {}
            for inter in interactions:
                if inter.weight > 1.0:
                    self.user_history.setdefault(inter.user_id, []).append(inter.item_id)
            
            adj_np = np.zeros((len(unique_users), len(unique_items)), dtype=np.float32)
            for inter in interactions:
                if inter.weight > 1.0 and inter.user_id in self.user_to_idx and inter.item_id in self.item_to_idx:
                    u_idx = self.user_to_idx[inter.user_id]
                    i_idx = self.item_to_idx[inter.item_id]
                    adj_np[u_idx, i_idx] = 1.0
                    
        else:
            interactions_path = os.path.join(data_dir, "interactions.csv")
            if not os.path.exists(interactions_path):
                raise FileNotFoundError(f"Missing interactions logs at: {interactions_path}")
                
            df = pd.read_csv(interactions_path)
            
            unique_users = df["user_id"].unique()
            unique_items = df["video_id"].unique()
            
            self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
            self.item_to_idx = {vid: idx for idx, vid in enumerate(unique_items)}
            self.idx_to_item = {idx: vid for vid, idx in self.item_to_idx.items()}
            
            for uid, group in df[df["click"] == 1].groupby("user_id"):
                self.user_history[uid] = group["video_id"].tolist()
                
            adj_np = np.zeros((len(unique_users), len(unique_items)), dtype=np.float32)
            for _, row in df[df["click"] == 1].iterrows():
                u_idx = self.user_to_idx[row["user_id"]]
                i_idx = self.item_to_idx[row["video_id"]]
                adj_np[u_idx, i_idx] = 1.0

        if adj_np.shape[0] == 0 or adj_np.shape[1] == 0:
            return

        self.adj_matrix = torch.tensor(adj_np, dtype=torch.float32).to(self.device)
        self.model = BipartiteGNNNet(
            num_users=len(self.user_to_idx),
            num_items=len(self.item_to_idx),
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            u_emb, i_emb = self.model(self.adj_matrix)
            pred_adj = torch.matmul(u_emb, i_emb.t())
            loss = nn.functional.binary_cross_entropy_with_logits(pred_adj, self.adj_matrix)
            loss.backward()
            optimizer.step()

    def recommend(self, user_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        if self.model is None or user_id not in self.user_to_idx:
            return []
            
        self.model.eval()
        u_idx = self.user_to_idx[user_id]
        history = self.user_history.get(user_id, [])
        
        with torch.no_grad():
            u_emb, i_emb = self.model(self.adj_matrix)
            user_vector = u_emb[u_idx].unsqueeze(0)
            scores = torch.matmul(user_vector, i_emb.t()).squeeze()
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            scores = scores.cpu().numpy()
            
        for vid in history:
            if vid in self.item_to_idx:
                scores[self.item_to_idx[vid]] = -1e5
                
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = []
        for idx in top_indices:
            vid = self.idx_to_item.get(idx, -1)
            if vid != -1:
                recommendations.append((vid, float(scores[idx])))
            
        return recommendations
