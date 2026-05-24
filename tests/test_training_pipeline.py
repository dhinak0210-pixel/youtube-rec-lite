import pytest
from training.pipeline import TrainingPipeline
from services.feature_store import FeatureStore
from config import settings

def test_training_pipeline_full_run():
    """Verifies that the complete TrainingPipeline runs cleanly in quick=True mode."""
    # Temporarily override settings scale to speed up test fittings
    settings.num_users = 100
    settings.num_items = 200
    settings.num_interactions = 500
    settings.mf_epochs = 1
    settings.bert_epochs = 1
    settings.gnn_epochs = 1
    settings.mmoe_epochs = 1
    
    pipeline = TrainingPipeline(num_users=100, num_items=200, num_interactions=500)
    
    # Run the full pipeline in quick=True mode
    results = pipeline.run(quick=True)
    
    # 1. Verify pipeline dictionary outputs
    assert "elapsed_seconds" in results
    assert "data_stats" in results
    assert "evaluation_results" in results
    
    # 2. Verify dataset limits got scaled in quick=True mode
    stats = results["data_stats"]
    assert stats["num_users"] <= 100
    assert stats["num_items"] <= 200
    assert stats["train_interactions"] > 0
    assert stats["test_interactions"] > 0
    
    # 3. Verify evaluation results structure
    eval_res = results["evaluation_results"]
    assert "CF" in eval_res
    assert "MF" in eval_res
    
    # 4. Verify feature store populated users and items
    assert pipeline.feature_store.get_user(1) is not None
    assert pipeline.feature_store.get_item(1) is not None
