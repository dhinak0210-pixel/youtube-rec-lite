"""
Central configuration settings module for the YouTube-scale recommendation system.

This module defines defaults for dataset scaling, hyper-parameters for matrix
factorization, sequential transformers (BERT4Rec), Relational GNNs, multi-task 
MMoE rankers, online streaming telemetry, and testing. Settings can be overridden
dynamically via environment variables using the prefix 'YTREC_'.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    num_users: int = 5000
    num_items: int = 10000  
    num_interactions: int = 100000
    num_categories: int = 20
    num_social_connections: int = 20000
    
    mf_num_factors: int = 32
    bert_d_model: int = 64
    bert_num_heads: int = 4
    bert_num_layers: int = 2
    bert_max_seq_len: int = 50
    gnn_embedding_dim: int = 32
    content_feature_dim: int = 16
    
    mf_epochs: int = 10
    bert_epochs: int = 5
    bert_batch_size: int = 128
    bert_lr: float = 0.001
    gnn_epochs: int = 5
    mmoe_epochs: int = 5
    mmoe_batch_size: int = 256
    
    candidate_pool_size: int = 200
    final_recommendation_size: int = 20
    cf_num_neighbors: int = 30
    cold_start_threshold: int = 5
    
    ab_min_samples: int = 100
    ab_significance_level: float = 0.05
    
    stream_window_seconds: int = 300
    model_version: str = "v1.0.0"
    use_fake_redis: bool = True
    log_level: str = "INFO"
    random_seed: int = 42

    model_config = SettingsConfigDict(
        env_prefix="YTREC_",
        case_sensitive=False
    )

# Singleton settings instance reference
settings = Settings()
