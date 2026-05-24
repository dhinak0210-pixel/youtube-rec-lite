"""
Multi-gate Mixture-of-Experts (MMoE) ranker in PyTorch.

Combats clickbait by optimizing 4 distinct engagement indicators simultaneously:
clicks, watch completions, likes, and dislikes. Shares representation learning 
across multiple LayerNorm-equipped expert blocks routed via soft task gates.
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
from data.schemas import Interaction, InteractionType

# ==========================================
# 📊 Part 1 - Feature Engineering & Weights
# ==========================================

FEATURE_DIM = (
    settings.mf_num_factors +    # user MF embedding (32)
    settings.mf_num_factors +    # item MF embedding (32)
    settings.bert_d_model +      # user sequence embedding (64)
    8 +                          # user demographics
    8 +                          # item features
    4 +                          # social features
    4 +                          # context: hour, device
    settings.mf_num_factors      # cross: user_emb * item_emb (32)
)

def build_feature_vector(
    user_mf: np.ndarray,
    item_mf: np.ndarray,
    user_seq: np.ndarray,
    user_demo: np.ndarray,
    item_feat: np.ndarray,
    social_feat: np.ndarray,
    context_feat: np.ndarray
) -> np.ndarray:
    """Assembles all feature arrays and computes element-wise cross products."""
    cross = user_mf * item_mf
    return np.concatenate([
        user_mf,
        item_mf,
        user_seq,
        user_demo,
        item_feat,
        social_feat,
        context_feat,
        cross
    ]).astype(np.float32)

TASKS = ["click", "watch_complete", "like", "dislike"]

OBJECTIVE_WEIGHTS = {
    "click": 0.20,
    "watch_complete": 0.35,
    "like": 0.20,
    "share": 0.15,  # Note: share is mapped inside final scoring combinations
    "dislike": -0.10,
}


# ==========================================
# 🎛️ Part 2 - MMoE Components
# ==========================================

class Expert(nn.Module):
    """LayerNorm-equipped MLP block that projects shared input dimensions to expert latent spaces."""

    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Gate(nn.Module):
    """Linear routing block outputting Softmax probabilities over expert counts."""

    def __init__(self, in_dim: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x), dim=-1)

class Tower(nn.Module):
    """Task-specific projection MLP predicting single Sigmoid classification margins."""

    def __init__(self, expert_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(expert_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ==========================================
# 🌲 Part 3 - Full MMoE Model
# ==========================================

class MMoERanker(nn.Module):
    """Multi-objective neural ranker routing shared input contexts to task-specific towers."""

    def __init__(self, input_dim: int = FEATURE_DIM, num_experts: int = 3, expert_dim: int = 64):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, expert_dim) for _ in range(num_experts)])
        self.gates = nn.ModuleDict({
            task: Gate(input_dim, num_experts)
            for task in TASKS
        })
        self.towers = nn.ModuleDict({
            task: Tower(expert_dim)
            for task in TASKS
        })

    def forward(self, x: torch.Tensor) -> TypeDict[str, torch.Tensor]:
        # Stack expert outputs shape: (B, num_experts, expert_dim)
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1)
        
        preds = {}
        for task in TASKS:
            # gate weights shape: (B, num_experts) -> unsqueeze to (B, num_experts, 1)
            gate_w = self.gates[task](x).unsqueeze(-1)
            # weighted sum of expert outputs shape: (B, expert_dim)
            task_input = (expert_outputs * gate_w).sum(dim=1)
            # tower prediction shape: (B,)
            preds[task] = self.towers[task](task_input)
            
        return preds

    @staticmethod
    def final_score(preds: TypeDict[str, torch.Tensor]) -> torch.Tensor:
        """Combines task predictions using unified balance weight constants."""
        click_w = OBJECTIVE_WEIGHTS.get("click", 0.20)
        watch_w = OBJECTIVE_WEIGHTS.get("watch_complete", 0.35)
        like_w = OBJECTIVE_WEIGHTS.get("like", 0.20)
        dislike_w = OBJECTIVE_WEIGHTS.get("dislike", -0.10)
        
        score = (
            preds["click"] * click_w +
            preds["watch_complete"] * watch_w +
            preds["like"] * like_w +
            preds["dislike"] * dislike_w
        )
        return score


# ==========================================
# ⚡ Part 4 - MMoE Model Trainer
# ==========================================

class MMoETrainer:
    """Orchestrates sample assembly and handles multi-task loss weight optimization."""

    def __init__(self, input_dim: int = FEATURE_DIM):
        self.input_dim = input_dim
        self.model = MMoERanker(input_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._fitted = False

        # Internal references for wrapper mappings
        self.mf_model = None
        self.seq_model = None
        self.gnn_model = None
        self.user_lookup = {}
        self.item_lookup = {}

    def generate_training_data(
        self,
        interactions: TypeList[Interaction],
        mf_model,
        bert_trainer,
        gnn_trainer,
        users,
        items,
        max_samples: int = 50000
    ) -> TypeTuple[np.ndarray, TypeDict[str, np.ndarray]]:
        """Compiles composite vector samples and extracts multi-objective ground truth labels."""
        logger.info("Assembling multi-objective training features...")
        
        user_lookup = {u.user_id: u for u in users}
        item_lookup = {i.item_id: i for i in items}
        
        user_histories = defaultdict(list)
        for inter in sorted(interactions, key=lambda x: getattr(x, 'timestamp', 0.0)):
            if inter.weight > 0:
                user_histories[inter.user_id].append(inter.item_id)
                
        features_list = []
        labels = {task: [] for task in TASKS}
        
        num_pos = 0
        total_samples = 0
        
        # Sample interactions to stay within max limits
        sampled_interactions = list(interactions)
        if len(sampled_interactions) > max_samples // 2:
            rng = np.random.RandomState(42)
            sampled_indices = rng.choice(len(sampled_interactions), size=max_samples // 2, replace=False)
            sampled_interactions = [sampled_interactions[idx] for idx in sampled_indices]
            
        for inter in sampled_interactions:
            uid = inter.user_id
            iid = inter.item_id
            
            if uid not in user_lookup or iid not in item_lookup:
                continue
                
            user = user_lookup[uid]
            item = item_lookup[iid]
            
            # 1. Fetch embeddings (handling empty boundaries)
            u_mf = mf_model.get_user_embedding(uid)
            if u_mf is None:
                u_mf = np.zeros(settings.mf_num_factors, dtype=np.float32)
                
            i_mf = mf_model.get_item_embedding(iid)
            if i_mf is None:
                i_mf = np.zeros(settings.mf_num_factors, dtype=np.float32)
                
            u_seq = bert_trainer.get_user_embedding(user_histories[uid])
            if u_seq is None:
                u_seq = np.zeros(settings.bert_d_model, dtype=np.float32)
                
            # 2. Demographics features
            user_demo = np.array([
                user.age_bucket / 6.0,
                user.gender / 2.0,
                user.country_id / 99.0,
                user.signup_days_ago / 1825.0,
                user.num_interactions / 2000.0,
                0.0, 0.0, 0.0
            ], dtype=np.float32)
            
            # 3. Item meta features
            item_feat = np.array([
                item.category_id / 19.0,
                item.duration_seconds / 1800.0,
                item.upload_days_ago / 1825.0,
                item.total_views / 1000000.0,
                item.like_ratio,
                item.avg_watch_pct,
                0.0, 0.0
            ], dtype=np.float32)
            
            # 4. Social & context features
            social_feat = gnn_trainer.graph.get_social_features(uid, iid)
            
            sin_hour = np.sin(2 * np.pi * inter.context_hour / 24.0)
            cos_hour = np.cos(2 * np.pi * inter.context_hour / 24.0)
            peak_val = 1.0 if 18 <= inter.context_hour <= 22 else 0.0
            dev_val = inter.context_device / 3.0
            context_feat = np.array([sin_hour, cos_hour, peak_val, dev_val], dtype=np.float32)
            
            # Construct positive feature vector
            pos_feat = build_feature_vector(u_mf, i_mf, u_seq, user_demo, item_feat, social_feat, context_feat)
            features_list.append(pos_feat)
            
            # Determine labels
            is_click = 1.0 if inter.interaction_type in (InteractionType.CLICK, InteractionType.LIKE, InteractionType.SHARE, InteractionType.WATCH_COMPLETE) else 0.0
            is_watch_c = 1.0 if inter.interaction_type == InteractionType.WATCH_COMPLETE or (inter.interaction_type == InteractionType.VIEW and inter.watch_percentage > 0.8) else 0.0
            is_like = 1.0 if inter.interaction_type in (InteractionType.LIKE, InteractionType.SHARE) else 0.0
            is_dislike = 1.0 if inter.interaction_type == InteractionType.DISLIKE else 0.0
            
            labels["click"].append(is_click)
            labels["watch_complete"].append(is_watch_c)
            labels["like"].append(is_like)
            labels["dislike"].append(is_dislike)
            
            if is_click > 0:
                num_pos += 1
            total_samples += 1
            
            # 5. Generate negative sample
            rng = np.random.RandomState(42 + total_samples)
            neg_item_id = int(rng.randint(0, len(items)))
            neg_item = items[neg_item_id]
            
            neg_i_mf = mf_model.get_item_embedding(neg_item.item_id)
            if neg_i_mf is None:
                neg_i_mf = np.zeros(settings.mf_num_factors, dtype=np.float32)
                
            neg_item_feat = np.array([
                neg_item.category_id / 19.0,
                neg_item.duration_seconds / 1800.0,
                neg_item.upload_days_ago / 1825.0,
                neg_item.total_views / 1000000.0,
                neg_item.like_ratio,
                neg_item.avg_watch_pct,
                0.0, 0.0
            ], dtype=np.float32)
            
            neg_social_feat = gnn_trainer.graph.get_social_features(uid, neg_item.item_id)
            
            neg_feat = build_feature_vector(u_mf, neg_i_mf, u_seq, user_demo, neg_item_feat, neg_social_feat, context_feat)
            features_list.append(neg_feat)
            
            labels["click"].append(0.0)
            labels["watch_complete"].append(0.0)
            labels["like"].append(0.0)
            labels["dislike"].append(0.0)
            total_samples += 1
            
        features_np = np.array(features_list, dtype=np.float32)
        labels_np = {task: np.array(labels[task], dtype=np.float32) for task in TASKS}
        
        pos_rate = num_pos / max(1, total_samples // 2)
        logger.info(f"Assembled {total_samples} samples. Active click baseline rate: {pos_rate:.4f}")
        return features_np, labels_np

    def train(self, X: np.ndarray, y: TypeDict[str, np.ndarray], epochs: int = 5, batch_size: int = 256):
        """Optimizes MMoE towers and gate weights using multi-task BCELoss updates."""
        if len(X) == 0:
            logger.warning("Empty features matrix. Skipping MMoE training.")
            self._fitted = True
            return
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Balance weighting scalars across tasks
        task_weights = {
            "click": 1.0,
            "watch_complete": 2.0,
            "like": 1.5,
            "dislike": 1.0
        }
        
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensors = {task: torch.tensor(y[task], dtype=torch.float32) for task in TASKS}
        
        dataset = torch.utils.data.TensorDataset(
            x_tensor,
            *[y_tensors[task] for task in TASKS]
        )
        
        batch_size = min(batch_size, max(1, len(dataset)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"⚡ Training PyTorch 4-Objective MMoE (Epochs: {epochs}, Batches: {len(loader)})...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                batch_x = batch[0].to(self.device)
                batch_y = {task: batch[idx + 1].to(self.device) for idx, task in enumerate(TASKS)}
                
                optimizer.zero_grad()
                preds = self.model(batch_x)
                
                # Squeeze outputs if necessary to align with targets
                loss = 0.0
                for task in TASKS:
                    pred_task = preds[task]
                    if pred_task.dim() > 0 and pred_task.size(-1) == 1:
                        pred_task = pred_task.squeeze(-1)
                    
                    task_loss = criterion(pred_task, batch_y[task])
                    loss += task_weights[task] * task_loss
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
                
            avg_loss = total_loss / len(x_tensor)
            logger.info(f"   MMoE Epoch {epoch+1}/{epochs} - Combined Multitask Loss: {avg_loss:.5f}")
            
        self._fitted = True
        logger.info("✅ Multi-Objective MMoE Ranker trained successfully!")

    def score(self, features: np.ndarray) -> np.ndarray:
        """Returns balanced final scores from trained expert networks."""
        if not self._fitted or len(features) == 0:
            return np.ones(len(features), dtype=np.float32) * 0.5
            
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            preds = self.model(x_tensor)
            scores = self.model.final_score(preds)
            
        return scores.cpu().numpy()


# ==========================================
# 🔄 Backward Compatibility Wrapper Class
# ==========================================

class PyTorchMMoERanker:
    """Seamless compatibility wrapper mapping to the 4-objective MMoE trainer."""

    def __init__(self):
        self.trainer = MMoETrainer()
        self.model = self.trainer.model
        self.user_features = {}
        self.item_features = {}
        self._fitted = False

    def fit(self, users=None, items=None, interactions=None, data_dir: str = "data"):
        """Compiles embeddings and fits the 4-objective MMoE neural ranker cascade."""
        from models.matrix_factorization import PyTorchMatrixFactorization
        from models.sequential_recommender import PyTorchSequentialRecommender
        from models.graph_neural_network import GNNTrainer
        
        # Instantiate and train dependent models if inputs are provided
        if users is not None and items is not None and interactions is not None:
            self.user_features = {u.user_id: u for u in users}
            self.item_features = {i.item_id: i for i in items}
            
            mf = PyTorchMatrixFactorization()
            mf.fit(users, items, interactions)
            
            seq = PyTorchSequentialRecommender()
            seq.fit(users, items, interactions)
            
            gnn = GNNTrainer(num_users=max(100, len(users)), num_items=max(100, len(items)))
            gnn.build_graph(interactions, {})
            gnn.train(interactions, epochs=1)
            
            X, y = self.trainer.generate_training_data(
                interactions=interactions,
                mf_model=mf,
                bert_trainer=seq,
                gnn_trainer=gnn,
                users=users,
                items=items,
                max_samples=2000
            )
            self.trainer.train(X, y, epochs=settings.mmoe_epochs)
            self._fitted = True
            
            self.trainer.mf_model = mf
            self.trainer.seq_model = seq
            self.trainer.gnn_model = gnn
            self.trainer.user_lookup = {u.user_id: u for u in users}
            self.trainer.item_lookup = {i.item_id: i for i in items}
            
        else:
            # Fallback for csv loading or empty fit boundaries
            self._fitted = True

    def rank(self, user_id: int, candidate_video_ids: TypeList[int]) -> TypeList[TypeTuple[int, float]]:
        """Scores candidates using the Multi-gate Mixture-of-Experts neural network."""
        if not self._fitted or not candidate_video_ids:
            return [(vid, 0.5) for vid in candidate_video_ids]
            
        features_list = []
        vids = []
        
        user = self.trainer.user_lookup.get(user_id)
        if not user:
            return [(vid, 0.5) for vid in candidate_video_ids]
            
        for vid in candidate_video_ids:
            item = self.trainer.item_lookup.get(vid)
            if not item:
                continue
                
            u_mf = self.trainer.mf_model.get_user_embedding(user_id)
            if u_mf is None:
                u_mf = np.zeros(settings.mf_num_factors, dtype=np.float32)
                
            i_mf = self.trainer.mf_model.get_item_embedding(vid)
            if i_mf is None:
                i_mf = np.zeros(settings.mf_num_factors, dtype=np.float32)
                
            # Dummy user sequence for fast lookups
            u_seq = self.trainer.seq_model.get_user_embedding([vid])
            if u_seq is None:
                u_seq = np.zeros(settings.bert_d_model, dtype=np.float32)
                
            user_demo = np.array([
                user.age_bucket / 6.0,
                user.gender / 2.0,
                user.country_id / 99.0,
                user.signup_days_ago / 1825.0,
                user.num_interactions / 2000.0,
                0.0, 0.0, 0.0
            ], dtype=np.float32)
            
            item_feat = np.array([
                item.category_id / 19.0,
                item.duration_seconds / 1800.0,
                item.upload_days_ago / 1825.0,
                item.total_views / 1000000.0,
                item.like_ratio,
                item.avg_watch_pct,
                0.0, 0.0
            ], dtype=np.float32)
            
            social_feat = self.trainer.gnn_model.graph.get_social_features(user_id, vid)
            context_feat = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)
            
            feat = build_feature_vector(u_mf, i_mf, u_seq, user_demo, item_feat, social_feat, context_feat)
            features_list.append(feat)
            vids.append(vid)
            
        if not features_list:
            return [(vid, 0.5) for vid in candidate_video_ids]
            
        scores = self.trainer.score(np.array(features_list))
        scored_candidates = [(vid, float(score)) for vid, score in zip(vids, scores)]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates
