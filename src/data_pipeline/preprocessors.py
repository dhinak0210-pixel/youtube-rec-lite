import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
from src.config import BERT_SEQ_LEN, PROCESSED_DATA_DIR
from src.utils.logger import logger

class RecommenderPreprocessor:
    """
    Transforms raw tabular data into numerical encodings, interaction sequences, 
    and graph adjacency mappings ready for deep learning models.
    """
    def __init__(self):
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.video_to_idx = {}
        self.idx_to_video = {}
        self.cat_to_idx = {}
        self.gender_to_idx = {}
        self.country_to_idx = {}

    def fit(self, users_df: pd.DataFrame, videos_df: pd.DataFrame):
        """
        Learns key mapping definitions from demographic and video catalogs.
        """
        logger.info("Fitting mappings on user and video metadata.")
        
        # User index mapping
        unique_users = users_df["user_id"].unique()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        
        # Video index mapping
        unique_videos = videos_df["video_id"].unique()
        # Reserve index 0 for Padding/Masking in sequence models like BERT4Rec
        self.video_to_idx = {vid: idx + 1 for idx, vid in enumerate(unique_videos)}
        self.idx_to_video = {idx + 1: vid for vid, idx in self.video_to_idx.items()}
        
        # Add a special token for padding (index 0)
        self.video_to_idx["PAD"] = 0
        self.idx_to_video[0] = "PAD"
        # Reserve the last index + 1 for [MASK] in BERT4Rec sequential model
        mask_idx = len(unique_videos) + 1
        self.video_to_idx["MASK"] = mask_idx
        self.idx_to_video[mask_idx] = "MASK"

        # Categorical labels
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(videos_df["category"].unique())}
        self.gender_to_idx = {g: idx for idx, g in enumerate(users_df["gender"].unique())}
        self.country_to_idx = {c: idx for idx, c in enumerate(users_df["country"].unique())}
        
        logger.info(f"Fitted: {len(self.user_to_idx)} Users, {len(self.video_to_idx) - 2} Videos (+PAD, +MASK).")

    def transform_metadata(self, users_df: pd.DataFrame, videos_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encodes categorical strings and raw ids into numerical sequences.
        """
        users_processed = users_df.copy()
        users_processed["user_idx"] = users_processed["user_id"].map(self.user_to_idx)
        users_processed["gender_idx"] = users_processed["gender"].map(self.gender_to_idx)
        users_processed["country_idx"] = users_processed["country"].map(self.country_to_idx)
        # Normalize continuous values
        users_processed["age_norm"] = users_processed["age"] / 100.0

        videos_processed = videos_df.copy()
        videos_processed["video_idx"] = videos_processed["video_id"].map(self.video_to_idx)
        videos_processed["category_idx"] = videos_processed["category"].map(self.cat_to_idx)
        videos_processed["duration_norm"] = videos_processed["duration"] / videos_processed["duration"].max()
        
        return users_processed, videos_processed

    def build_sequential_data(self, interactions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates sequence history input matrices for BERT4Rec (Sequential).
        """
        logger.info(f"Building sequential interaction windows with length {BERT_SEQ_LEN}.")
        
        # Map IDs to indices
        df = interactions_df.copy()
        df = df[df["click"] == 1]  # Model sequential interest over positive clicks
        df = df.sort_values(by="timestamp")
        df["user_idx"] = df["user_id"].map(self.user_to_idx)
        df["video_idx"] = df["video_id"].map(self.video_to_idx)
        
        # Group by user to extract historic sequence
        user_history = df.groupby("user_idx")["video_idx"].apply(list).to_dict()
        
        sequences = []
        targets = []
        
        for user_idx, history in user_history.items():
            if len(history) < 2:
                continue
            
            # Create rolling sub-sequences
            for i in range(1, len(history)):
                seq = history[:i]
                target = history[i]
                
                # Truncate or Pad sequence (left pad with 0s)
                if len(seq) > BERT_SEQ_LEN:
                    seq = seq[-BERT_SEQ_LEN:]
                else:
                    seq = [0] * (BERT_SEQ_LEN - len(seq)) + seq
                    
                sequences.append(seq)
                targets.append(target)
                
        return np.array(sequences), np.array(targets)

    def build_graph_adjacency(self, interactions_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Edge list representations for GNN connecting Users and Videos.
        """
        df = interactions_df.copy()
        df = df[df["click"] == 1]
        
        user_indices = df["user_id"].map(self.user_to_idx).values
        video_indices = df["video_id"].map(self.video_to_idx).values
        
        offset = len(self.video_to_idx)
        
        # Bidirectional graph edge indices
        u_nodes = user_indices + offset
        v_nodes = video_indices
        
        edge_start = np.concatenate([u_nodes, v_nodes])
        edge_end = np.concatenate([v_nodes, u_nodes])
        
        edge_index = torch.tensor(np.stack([edge_start, edge_end]), dtype=torch.long)
        logger.info(f"Graph initialized with {edge_index.shape[1]} bidirectional interaction edges.")
        return edge_index, torch.tensor(v_nodes, dtype=torch.long)

    def build_ranking_features(self, 
                               interactions_df: pd.DataFrame, 
                               users_processed: pd.DataFrame, 
                               videos_processed: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares training features for the MMoE Ranking engine:
        - X: Combined features [user_idx, gender_idx, country_idx, age_norm, video_idx, category_idx, duration_norm]
        - y_click: Click objective labels
        - y_watch: Watch ratio objective labels (continuous regression)
        """
        logger.info("Constructing combined ranking features for MMoE model.")
        
        df = interactions_df.merge(users_processed, on="user_id", how="left")
        df = df.merge(videos_processed, on="video_id", how="left")
        
        # Calculate watch_ratio from watch_percentage
        df["watch_ratio"] = df["watch_percentage"] / 100.0
        
        # Column selections including gender and country
        feature_cols = [
            "user_idx", "gender_idx", "country_idx", "age_norm", 
            "video_idx", "category_idx", "duration_norm"
        ]
        
        # Clean any possible NaNs
        df = df.dropna(subset=feature_cols + ["click", "watch_ratio"])
        
        X = df[feature_cols].values
        y_click = df["click"].values.astype(np.float32)
        y_watch = df["watch_ratio"].values.astype(np.float32)
        
        return X, y_click, y_watch
