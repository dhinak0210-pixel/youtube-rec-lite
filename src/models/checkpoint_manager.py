import torch
import os
from pathlib import Path
from src.utils.logger import logger

def get_checkpoint_path(checkpoint_dir: Path) -> Path:
    """Returns the path to the latest checkpoint file."""
    return checkpoint_dir / "recosystem_checkpoint.pt"

def save_checkpoint(
    checkpoint_dir: Path,
    preprocessor,
    model_cf,
    model_als,
    model_bert,
    model_gnn,
    model_mmoe,
    cold_start_handler,
    social_graph
) -> bool:
    """
    Saves the entire recommendation engine state, including preprocessors,
    retrieval models, ranking engine, cold start maps, and the social graph.
    """
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = get_checkpoint_path(checkpoint_dir)
        
        logger.info(f"Saving system checkpoint to {checkpoint_path}...")
        
        checkpoint_dict = {
            "preprocessor": preprocessor,
            "model_cf": model_cf,
            "model_als": model_als,
            "model_bert": model_bert,
            "model_gnn": model_gnn,
            "model_mmoe": model_mmoe,
            "cold_start_handler": cold_start_handler,
            "social_graph": social_graph
        }
        
        torch.save(checkpoint_dict, checkpoint_path)
        logger.info("✅ Checkpoint saved successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {str(e)}")
        return False

def load_checkpoint(checkpoint_dir: Path):
    """
    Loads and returns the recommendation engine state if a checkpoint exists.
    Returns:
        dict if loaded successfully, None otherwise.
    """
    checkpoint_path = get_checkpoint_path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.info(f"No checkpoint found at {checkpoint_path}. A new model must be trained.")
        return None
        
    try:
        logger.info(f"Loading system checkpoint from {checkpoint_path}...")
        # Load tensors on CPU to avoid device mapping conflicts; set weights_only=False to support custom classes
        checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        
        # Verify that all expected keys exist in the loaded dictionary
        required_keys = [
            "preprocessor", "model_cf", "model_als", "model_bert",
            "model_gnn", "model_mmoe", "cold_start_handler", "social_graph"
        ]
        for key in required_keys:
            if key not in checkpoint_dict:
                raise KeyError(f"Missing required key '{key}' in checkpoint.")
                
        logger.info("✅ Checkpoint loaded successfully!")
        return checkpoint_dict
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {str(e)}")
        return None
