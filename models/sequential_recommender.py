"""
Transformer-based Sequential Recommender model (inspired by BERT4Rec).

Models user watch history logs as ordered sequences, learning dynamic
shifting interest representations using positional embeddings and Multi-Head
Attention Transformer encoder blocks to predict next-video candidates.
"""

import os
import math
from typing import List as TypeList, Set as TypeSet, Optional
import numpy as np
import pandas as pd
from loguru import logger
import torch
import torch.nn as nn
from config import settings
from data.schemas import Interaction, ScoredItem

# ==========================================
# 🧱 Part 1 - Building Blocks
# ==========================================

class PositionalEncoding(nn.Module):
    """Adds sinusoids to model sequence order metrics without recurrent structures."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Calculates scaled dot-product attention across multiple linear subspaces in parallel."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        
        # Project and reshape: (B, T, d_model) -> (B, num_heads, T, head_dim)
        q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Context pooling
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.Wo(context)

class TransformerBlock(nn.Module):
    """Standard Transformer Encoder Layer using MultiHeadAttention and FeedForward blocks."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Attention residual path
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_out))
        # Feed-Forward network residual path
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ==========================================
# 🧠 Part 2 - Main Model
# ==========================================

class BERT4Rec(nn.Module):
    """BERT4Rec Sequence Encoder using Bidirectional Multi-Head Self-Attention layers."""
    MASK_ID = 0

    def __init__(
        self,
        num_items: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # +1 because item 0 is reserved for MASK/PAD tokens
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len + 1)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_items + 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq shape: (B, T)
        x = self.item_emb(seq)
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        logits = self.head(x)  # shape: (B, T, num_items + 1)
        return logits

    def get_sequence_embedding(self, seq: torch.Tensor) -> torch.Tensor:
        # seq shape: (B, T)
        x = self.item_emb(seq)
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        # Return last position output (before projection head)
        return x[:, -1, :]


# ==========================================
# ⚡ Part 3 - Trainer Wrapper
# ==========================================

class BERT4RecTrainer:
    """Supervises sequential training datasets using mask filling cross entropy optimization."""

    def __init__(self, num_items: int):
        """Initializes the underlying BERT4Rec network architecture."""
        self.num_items = num_items
        self.d_model = settings.bert_d_model
        self.nhead = settings.bert_num_heads
        self.num_layers = settings.bert_num_layers
        self.max_seq_len = settings.bert_max_seq_len
        self.epochs = settings.bert_epochs
        
        self.model = BERT4Rec(
            num_items=num_items,
            d_model=self.d_model,
            num_heads=self.nhead,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._fitted = False

    def build_sequences(self, interactions: TypeList[Interaction], min_len: int = 3) -> dict[int, list[int]]:
        """Sorts interactions chronologically per user and shifts item IDs to 1-indexed (0 reserved)."""
        # Sort chronologically by timestamp
        sorted_inters = sorted(interactions, key=lambda x: x.timestamp)
        
        user_to_vids = {}
        for inter in sorted_inters:
            if 0 <= inter.item_id < self.num_items:
                user_to_vids.setdefault(inter.user_id, []).append(inter.item_id + 1)
                
        # Filter to sequences matching minimum length criteria
        filtered_sequences = {
            uid: vids for uid, vids in user_to_vids.items()
            if len(vids) >= min_len
        }
        
        return filtered_sequences

    def train(self, interactions: TypeList[Interaction], epochs: int = None, batch_size: int = 128, mask_prob: float = 0.20) -> list[float]:
        """Trains the sequential network using masked language model updates."""
        if epochs is None:
            epochs = self.epochs
            
        logger.info("Training BERT4Rec Sequential Transformer...")
        
        # 1. Build sequences from interactions
        user_sequences = self.build_sequences(interactions)
        if not user_sequences:
            logger.warning("No user sequences satisfied the minimum length requirement. Skipping training.")
            self._fitted = True
            return []
            
        sequences_list = list(user_sequences.values())
        
        # 2. Setup training parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=settings.bert_lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            # Shuffle sequences
            indices = np.arange(len(sequences_list))
            np.random.shuffle(indices)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for start_idx in range(0, len(sequences_list), batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                batch_seqs = [sequences_list[idx] for idx in batch_indices]
                
                # Mask batch sequence values
                inputs, targets = self._prepare_batch(batch_seqs, mask_prob)
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Logits shape: (B, T, num_items + 1)
                logits = self.model(inputs)
                
                # Reshape for cross entropy loss
                loss = criterion(logits.view(-1, self.num_items + 1), targets.view(-1))
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / max(1, num_batches)
            losses.append(avg_loss)
            logger.info(f"   BERT4Rec Epoch {epoch+1}/{epochs} - Mask Reconstruction Loss: {avg_loss:.5f}")
            
        self._fitted = True
        logger.info("BERT4Rec training completed successfully!")
        return losses

    def _prepare_batch(self, seqs: list[list[int]], mask_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncates, pads left with zeros, and masks token vectors with target ignore margins."""
        batch_inputs = []
        batch_targets = []
        
        for seq in seqs:
            # Truncate
            truncated = seq[-self.max_seq_len:]
            # Pad left with 0
            padded = [0] * (self.max_seq_len - len(truncated)) + truncated
            
            inputs = []
            targets = []
            
            for token in padded:
                if token == 0:
                    inputs.append(0)
                    targets.append(-100) # Ignored
                else:
                    if np.random.rand() < mask_prob:
                        inputs.append(self.model.MASK_ID)
                        targets.append(token) # Predict target
                    else:
                        inputs.append(token)
                        targets.append(-100) # Ignored
                        
            # Enforce at least one masked index to calculate loss gradients
            if all(t == -100 for t in targets) and truncated:
                # Find a random valid padded index (non-zero padding)
                active_indexes = [idx for idx, val in enumerate(padded) if val != 0]
                if active_indexes:
                    masked_idx = np.random.choice(active_indexes)
                    inputs[masked_idx] = self.model.MASK_ID
                    targets[masked_idx] = padded[masked_idx]

            batch_inputs.append(inputs)
            batch_targets.append(targets)
            
        return torch.tensor(batch_inputs, dtype=torch.long), torch.tensor(batch_targets, dtype=torch.long)

    def predict(self, user_history: list[int], top_k: int = 100, exclude: TypeSet[int] = None) -> list[ScoredItem]:
        """Predicts recommendation candidate list by computing probability distributions over MASK logits."""
        if not self._fitted or not user_history:
            return []

        self.model.eval()
        
        # Shift to 1-indexed
        history_shifted = [vid + 1 for vid in user_history if 0 <= vid < self.num_items]
        if not history_shifted:
            return []

        # Truncate to max_seq_len - 1, and append MASK (0) at the end
        truncated = history_shifted[-(self.max_seq_len - 1):] + [self.model.MASK_ID]
        # Pad left with zeros
        padded = [0] * (self.max_seq_len - len(truncated)) + truncated
        
        with torch.no_grad():
            seq_tensor = torch.tensor([padded], dtype=torch.long).to(self.device)
            # Logits shape: (1, T, num_items + 1)
            logits = self.model(seq_tensor)
            
            # Extract last position logits (representing the MASK token output logits)
            last_logits = logits[0, -1, 1:]  # Exclude padding token (idx 0)
            scores = torch.softmax(last_logits, dim=-1).cpu().numpy()

        # Compile exclusions
        exclude_set = set(user_history)
        if exclude:
            exclude_set.update(exclude)
            
        for vid in exclude_set:
            if 0 <= vid < len(scores):
                scores[vid] = -np.inf
                
        # Sort and return top_k candidates
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if np.isneginf(score):
                continue
            results.append(ScoredItem(item_id=int(idx), score=score, source="sequential"))
            
        return results

    def get_user_embedding(self, history: list[int]) -> Optional[np.ndarray]:
        """Retrieves seq output representation array (d_model dim) for user history."""
        if not self._fitted or not history:
            return None

        self.model.eval()
        history_shifted = [vid + 1 for vid in history if 0 <= vid < self.num_items]
        if not history_shifted:
            return None

        truncated = history_shifted[-self.max_seq_len:]
        padded = [0] * (self.max_seq_len - len(truncated)) + truncated
        
        with torch.no_grad():
            seq_tensor = torch.tensor([padded], dtype=torch.long).to(self.device)
            embedding = self.model.get_sequence_embedding(seq_tensor).squeeze()
            return embedding.cpu().numpy().copy()


# ==========================================
# 🔄 Backward Compatibility Legacy Class
# ==========================================

class TransformerSeqNet(nn.Module):
    def __init__(self, num_items: int, d_model: int, nhead: int, num_layers: int, max_len: int):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_items)
        
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = seqs.size()
        pos = torch.arange(seq_len, dtype=torch.long, device=seqs.device).unsqueeze(0).repeat(batch_size, 1)
        x = self.item_embedding(seqs) + self.pos_embedding(pos)
        padding_mask = (seqs == 0)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        logits = self.fc(pooled)
        return logits

class PyTorchSequentialRecommender:
    def __init__(self):
        self.d_model = settings.bert_d_model
        self.nhead = settings.bert_num_heads
        self.num_layers = settings.bert_num_layers
        self.max_len = settings.bert_max_seq_len
        self.epochs = settings.bert_epochs
        self.batch_size = settings.bert_batch_size
        self.lr = settings.bert_lr
        self.model = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_sequences = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, users=None, items=None, interactions=None, data_dir: str = "data"):
        if users is not None and items is not None and interactions is not None:
            unique_items = sorted(list(set(i.item_id for i in items)))
            self.item_to_idx = {vid: (idx + 1) for idx, vid in enumerate(unique_items)}
            self.idx_to_item = {idx: vid for vid, idx in self.item_to_idx.items()}
            sorted_inters = sorted([i for i in interactions if i.weight > 1.0], key=lambda x: x.timestamp)
            
            self.user_sequences = {}
            for inter in sorted_inters:
                if inter.item_id in self.item_to_idx:
                    self.user_sequences.setdefault(inter.user_id, []).append(self.item_to_idx[inter.item_id])
            
            sequences_list = []
            target_list = []
            for uid, vids in self.user_sequences.items():
                if len(vids) < 3:
                    continue
                for i in range(1, len(vids)):
                    seq = vids[max(0, i - self.max_len):i]
                    target = vids[i] - 1
                    padded_seq = [0] * (self.max_len - len(seq)) + seq
                    sequences_list.append(padded_seq)
                    target_list.append(target)
        else:
            interactions_path = os.path.join(data_dir, "interactions.csv")
            if not os.path.exists(interactions_path):
                raise FileNotFoundError(f"Missing interactions at: {interactions_path}")
            df = pd.read_csv(interactions_path)
            click_df = df[df["click"] == 1].sort_values(by="timestamp")
            unique_items = df["video_id"].unique()
            self.item_to_idx = {vid: (idx + 1) for idx, vid in enumerate(unique_items)}
            self.idx_to_item = {idx: vid for vid, idx in self.item_to_idx.items()}
            
            sequences_list = []
            target_list = []
            for uid, group in click_df.groupby("user_id"):
                vids = [self.item_to_idx[vid] for vid in group["video_id"] if vid in self.item_to_idx]
                self.user_sequences[uid] = vids
                if len(vids) < 3:
                    continue
                for i in range(1, len(vids)):
                    seq = vids[max(0, i - self.max_len):i]
                    target = vids[i] - 1
                    padded_seq = [0] * (self.max_len - len(seq)) + seq
                    sequences_list.append(padded_seq)
                    target_list.append(target)

        if not sequences_list:
            return
            
        seq_tensor = torch.tensor(sequences_list, dtype=torch.long)
        target_tensor = torch.tensor(target_list, dtype=torch.long)
        self.model = TransformerSeqNet(
            num_items=len(self.item_to_idx),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            max_len=self.max_len
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(seq_tensor, target_tensor)
        batch_size = min(self.batch_size, max(1, len(dataset)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for seqs, targets in loader:
                seqs, targets = seqs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(seqs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def recommend(self, user_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        if self.model is None or user_id not in self.user_sequences:
            return []
        self.model.eval()
        vids = self.user_sequences[user_id]
        if not vids:
            return []
        seq = vids[-self.max_len:]
        padded_seq = [0] * (self.max_len - len(seq)) + seq
        
        with torch.no_grad():
            seq_tensor = torch.tensor([padded_seq], dtype=torch.long).to(self.device)
            logits = self.model(seq_tensor)
            scores = torch.softmax(logits, dim=1).squeeze()
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            scores = scores.cpu().numpy()
            
        history_set = set(vids)
        for idx_val in range(len(scores)):
            vid = self.idx_to_item.get(idx_val + 1, -1)
            if vid in history_set:
                scores[idx_val] = -1.0
                
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = []
        for idx in top_indices:
            vid = self.idx_to_item.get(idx + 1, -1)
            if vid != -1 and scores[idx] > 0.0:
                recommendations.append((vid, float(scores[idx])))
        return recommendations

    def get_user_embedding(self, history: list[int]) -> Optional[np.ndarray]:
        """Retrieves seq output representation array (d_model dim) for user history."""
        if self.model is None or not history:
            return None
        self.model.eval()
        history_shifted = [self.item_to_idx[vid] for vid in history if vid in self.item_to_idx]
        if not history_shifted:
            return None
        truncated = history_shifted[-self.max_len:]
        padded = [0] * (self.max_len - len(truncated)) + truncated
        with torch.no_grad():
            seqs = torch.tensor([padded], dtype=torch.long).to(self.device)
            batch_size, seq_len = seqs.size()
            pos = torch.arange(seq_len, dtype=torch.long, device=seqs.device).unsqueeze(0).repeat(batch_size, 1)
            x = self.model.item_embedding(seqs) + self.model.pos_embedding(pos)
            padding_mask = (seqs == 0)
            x = self.model.transformer(x, src_key_padding_mask=padding_mask)
            mask = (~padding_mask).unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            emb = pooled.squeeze(0).cpu().numpy()
        return emb

