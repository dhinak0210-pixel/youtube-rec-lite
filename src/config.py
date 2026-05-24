import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"

# Create directories if they do not exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Simulation Parameters
NUM_USERS = 1000
NUM_VIDEOS = 500
NUM_CATEGORIES = 10
SIMULATION_STEPS = 5000  # Number of initial historical interactions

# Candidate Generation Parameters
TOP_K_CANDIDATES = 50  # Number of candidates retrieved per user
EMBEDDING_DIM = 64     # Standard embedding size across MF, BERT, GNN

# BERT4Rec Model Hyperparameters
BERT_SEQ_LEN = 10      # Length of user interaction sequence
BERT_NUM_HEADS = 2
BERT_NUM_LAYERS = 2
BERT_DROPOUT = 0.1

# GNN Parameters
GNN_NUM_EPOCHS = 10
GNN_LR = 0.01

# MMoE Ranking Model Hyperparameters
MMOE_NUM_EXPERTS = 3
MMOE_EXPERT_HIDDEN = 64
MMOE_TOWER_HIDDEN = 32

# Redis & API Ports
REDIS_HOST = "localhost"
REDIS_PORT = 6379
API_HOST = "127.0.0.1"
API_PORT = 8010
UI_PORT = 7870
