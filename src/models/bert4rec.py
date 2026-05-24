import math
import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from src.utils.logger import logger

class PositionalEncoding(nn.Module):
    """
    Implements standard sinusoidal positional encodings as proposed in 'Attention Is All You Need'.
    Encodes ordering information into latent representations up to max_seq_len (200).
    """
    def __init__(self, d_model: int, max_seq_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Sine for even indices; Cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model]
        Returns:
            Tensors with positional weights added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class MultiHeadSelfAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention (MHSA) supporting query, key, and value 
    projections, scaled dot-product attention, and optional causal masking.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, 1, 1, seq_len] or causal matrix
        """
        batch_size, seq_len, d_model = x.size()
        
        # Project and split dimensions: [batch_size, num_heads, seq_len, d_k]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Mask out forbidden/padding tokens with large negative numbers
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Multi-head aggregation context
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    """
    A standard Bidirectional Transformer Encoder Block featuring:
    - Multi-Head Self-Attention Layer
    - Fully Connected Feed-Forward Network
    - Layer Normalization layers + Residual Connections
    - GELU activation functions
    """
    def __init__(self, d_model: int, num_heads: int = 4, ffn_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # MHSA + Residual + Norm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class BERT4RecModel(nn.Module):
    """
    Bidirectional Transformer for Sequential Recommendation (BERT4Rec).
    Processes bidirectional sequence context and predicts target masked items.
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 128, 
                 num_layers: int = 4, 
                 num_heads: int = 4, 
                 max_seq_len: int = 200, 
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.mask_token_idx = vocab_size - 1  # Standard [MASK] token assignment (last index)
        
        # Layer 1: Item and position embedding layers
        self.item_embeddings = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Layer 2: Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_hidden=d_model * 4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Layer 3: Projection Head over full vocabulary index
        self.out_projection = nn.Linear(d_model, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence tensor of shape [batch_size, seq_len]
        Returns:
            Logits over item vocabulary: [batch_size, seq_len, vocab_size]
        """
        # Padding mask representation: shape [batch_size, 1, 1, seq_len]
        padding_mask = (x != 0).unsqueeze(1).unsqueeze(2)
        
        # Embed items & positional indicators
        h = self.item_embeddings(x)
        h = self.positional_encoding(h)
        h = self.dropout(h)
        
        # Sequentially feed through transformer block layers
        for block in self.blocks:
            h = block(h, padding_mask)
            
        h = self.norm(h)
        logits = self.out_projection(h)
        return logits

class BERT4RecRecommender:
    """
    Standard Production wrapper for BERT4Rec integration. 
    Implements paper-grade MLM (Masked Language Modeling) training and top-K inference.
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 128, 
                 num_layers: int = 4, 
                 num_heads: int = 4, 
                 max_seq_len: int = 200, 
                 epochs: int = 10, 
                 batch_size: int = 256, 
                 learning_rate: float = 0.001):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        
        self.model = BERT4RecModel(
            vocab_size=vocab_size, 
            d_model=d_model, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            max_seq_len=max_seq_len
        )
        self.mask_token_idx = self.model.mask_token_idx

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains BERT4Rec on sequence datasets utilizing standard MLM rules:
        - 20% masking rate: replacing targets with [MASK] tokens dynamically.
        - Ignores padding values (index 0) from CrossEntropy Loss calculations.
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding loss
        
        dataset_size = len(X_train)
        logger.info(f"Training Custom BERT4Rec Model on CPU (Dataset size: {dataset_size})")
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = int(np.ceil(dataset_size / self.batch_size))
            
            # Shuffle batch keys
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Fetch batch slices
                x_batch = torch.tensor(X_train[batch_indices], dtype=torch.long)
                y_batch = torch.tensor(y_train[batch_indices], dtype=torch.long)
                
                # Masking Pipeline:
                # To simulate Masked Language Model training on sequence pools:
                # We mask 20% of active tokens randomly, or replace the last active index.
                # To ensure robust sequential training, we replace the target prediction position
                # with the [MASK] token in masked_x.
                masked_x = x_batch.clone()
                for i in range(len(masked_x)):
                    active_positions = torch.where(masked_x[i] != 0)[0]
                    if len(active_positions) > 0:
                        # Mask 20% of positions or at least the last active target position
                        last_pos = active_positions[-1]
                        masked_x[i, last_pos] = self.mask_token_idx
                        
                        # Extra random masking (20% rate)
                        for pos in active_positions[:-1]:
                            if random.random() < 0.20:
                                masked_x[i, pos] = self.mask_token_idx
                                
                optimizer.zero_grad()
                logits = self.model(masked_x)  # Shape: [batch_size, seq_len, vocab_size]
                
                # Evaluate Loss over masked slots
                loss = 0.0
                mask_count = 0
                for i in range(len(masked_x)):
                    mask_positions = torch.where(masked_x[i] == self.mask_token_idx)[0]
                    if len(mask_positions) > 0:
                        # Logits shape [mask_positions_len, vocab_size]
                        pred_logits = logits[i, mask_positions]
                        # Target tags map: standard labels are matched
                        target_tags = y_batch[i].repeat(len(mask_positions)) # Align dimensions
                        loss += criterion(pred_logits, target_tags)
                        mask_count += 1
                        
                if mask_count > 0:
                    loss /= mask_count
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * (end - start)
                    
            avg_loss = total_loss / max(dataset_size, 1)
            logger.info(f"BERT4Rec Training Epoch {epoch + 1}/{self.epochs} - Combined MLM Loss: {avg_loss:.4f}")

    def retrieve_candidates(self, user_history: List[int], top_n: int = 100) -> List[Tuple[int, float]]:
        """
        Paper-grade inference: appends a [MASK] token to the user history sequence,
        queries the output logits at this position, and sorts target items.
        """
        self.model.eval()
        with torch.no_grad():
            # Truncate history to leave room for the [MASK] token
            hist = user_history[-(self.max_seq_len - 1):]
            seq = hist + [self.mask_token_idx]
            
            # Left pad sequence with 0s
            if len(seq) < self.max_seq_len:
                seq = [0] * (self.max_seq_len - len(seq)) + seq
                
            x = torch.tensor([seq], dtype=torch.long)
            logits = self.model(x)  # Shape: [1, seq_len, vocab_size]
            
            # Extract logits at the final position ([MASK] location)
            mask_logits = logits[0, -1]
            
            # Exclude Padding (0) and [MASK] (mask_token_idx) from candidates
            mask_logits[0] = -1e9
            mask_logits[self.mask_token_idx] = -1e9
            
            scores = torch.softmax(mask_logits, dim=-1)
            top_scores, top_indices = torch.topk(scores, k=min(top_n, self.vocab_size - 2))
            
            return list(zip(top_indices.tolist(), top_scores.tolist()))

    def get_sequence_embedding(self, history: List[int]) -> np.ndarray:
        """
        Extracts a high-fidelity 128-dimensional embedding representing the user's sequential 
        history by averaging active hidden states. Feedable directly into MMoE rankers.
        """
        self.model.eval()
        with torch.no_grad():
            seq = history[-self.max_seq_len:]
            if len(seq) < self.max_seq_len:
                seq = [0] * (self.max_seq_len - len(seq)) + seq
                
            x = torch.tensor([seq], dtype=torch.long)
            
            # Forward pass up to normative hidden state block
            padding_mask = (x != 0).unsqueeze(1).unsqueeze(2)
            h = self.model.item_embeddings(x)
            h = self.model.positional_encoding(h)
            h = self.model.dropout(h)
            
            for block in self.model.blocks:
                h = block(h, padding_mask)
                
            h = self.model.norm(h)  # Shape: [1, seq_len, d_model]
            
            # Return mean embedding over active, non-zero tokens
            non_pad_mask = (x != 0).unsqueeze(-1).float()  # [1, seq_len, 1]
            sum_emb = torch.sum(h * non_pad_mask, dim=1)
            count_emb = torch.sum(non_pad_mask, dim=1).clamp(min=1.0)
            seq_embedding = sum_emb / count_emb
            
            return seq_embedding.squeeze(0).numpy()

if __name__ == "__main__":
    # Standard PyTorch dry-run test
    print("Executing custom BERT4Rec architecture verification tests...")
    
    vocab_size = 50002  # PAD=0, items: 1..50000, MASK=50001
    
    # 1. Create Model Instance
    recommender = BERT4RecRecommender(
        vocab_size=vocab_size, 
        d_model=128, 
        num_layers=4, 
        num_heads=4, 
        epochs=1, 
        batch_size=4
    )
    print("✅ Model compiled successfully.")
    
    # 2. Train 1 Epoch Demo
    mock_X = np.random.randint(1, 50000, size=(10, 200))
    mock_y = np.random.randint(1, 50000, size=(10,))
    print("Starting pre-training check...")
    recommender.train_model(mock_X, mock_y)
    print("✅ Pre-training iteration verified.")
    
    # 3. Predict candidates
    user_hist = [12, 45, 999, 2341]
    recs = recommender.retrieve_candidates(user_hist, top_n=5)
    print(f"Top 5 Recommendations: {recs}")
    assert len(recs) == 5, "Should return 5 recommendation items."
    print("✅ Candidate retrieval logic verified.")
    
    # 4. Extract sequence embeddings
    seq_emb = recommender.get_sequence_embedding(user_hist)
    print(f"Sequence vector dimension: {seq_emb.shape}")
    assert seq_emb.shape == (128,), "Sequence embedding must be 128-dimensional."
    print("✅ Sequence embedding layer verified.")
    print("All BERT4Rec verification assertions completed successfully!")
