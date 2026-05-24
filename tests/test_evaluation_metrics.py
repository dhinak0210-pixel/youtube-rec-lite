import pytest
import numpy as np
from evaluation.metrics import Metrics

def test_precision_at_k():
    """Verifies precision_at_k metric math bounds."""
    # Top-3 precision: 1 hit out of 3 -> 1/3 = 0.3333
    pred = [1, 2, 3, 4]
    rel = {2, 4, 5}
    p = Metrics.precision_at_k(pred, rel, k=3)
    assert abs(p - 1/3) < 1e-6
    
    p = Metrics.precision_at_k(pred, rel, k=4)
    assert abs(p - 0.5) < 1e-6

    # Boundary checks
    assert Metrics.precision_at_k([], rel, k=3) == 0.0
    assert Metrics.precision_at_k(pred, rel, k=0) == 0.0
    assert Metrics.precision_at_k(pred, rel, k=-1) == 0.0

def test_recall_at_k():
    """Verifies recall_at_k metric math bounds."""
    # Top-3 recall: 1 hit (item 2) out of 3 total relevant -> 1/3 = 0.3333
    pred = [1, 2, 3, 4]
    rel = {2, 4, 5}
    r = Metrics.recall_at_k(pred, rel, k=3)
    assert abs(r - 1/3) < 1e-6
    
    r = Metrics.recall_at_k(pred, rel, k=4)
    assert abs(r - 2/3) < 1e-6

    # Boundary checks
    assert Metrics.recall_at_k(pred, set(), k=3) == 0.0
    assert Metrics.recall_at_k([], rel, k=3) == 0.0
    assert Metrics.recall_at_k(pred, rel, k=0) == 0.0

def test_ndcg_at_k():
    """Verifies ndcg_at_k metric calculations against manual values."""
    pred = [1, 2, 3]
    rel = {2, 4}
    
    # DCG@2: 0/log2(2) + 1/log2(3) = 1 / 1.58496 = 0.63093
    # IDCG@2: 1/log2(2) + 1/log2(3) = 1.63093
    # NDCG@2: 0.63093 / 1.63093 = 0.38685
    n = Metrics.ndcg_at_k(pred, rel, k=2)
    assert abs(n - 0.3868528) < 1e-5

    # Boundary checks
    assert Metrics.ndcg_at_k([], rel, k=2) == 0.0
    assert Metrics.ndcg_at_k(pred, set(), k=2) == 0.0

def test_hit_rate():
    """Verifies hit_rate flags."""
    pred = [1, 2, 3]
    rel = {2, 4}
    
    assert Metrics.hit_rate(pred, rel, k=2) == 1.0
    assert Metrics.hit_rate(pred, rel, k=1) == 0.0
    assert Metrics.hit_rate([], rel, k=2) == 0.0

def test_coverage():
    """Verifies catalog coverage calculations."""
    all_recs = [[1, 2], [2, 3], [1, 3]]
    # 3 unique recommended items out of 5 catalog items -> 3/5 = 0.6
    assert abs(Metrics.coverage(all_recs, total_items=5) - 0.6) < 1e-6
    assert Metrics.coverage(all_recs, total_items=0) == 0.0

def test_diversity():
    """Verifies category diversity calculations."""
    rec = [1, 2, 3]
    item_cats = {1: 10, 2: 10, 3: 20}
    # 2 unique categories represented by 3 items -> 2/3 = 0.6667
    assert abs(Metrics.diversity(rec, item_cats) - 2/3) < 1e-6
    assert Metrics.diversity([], item_cats) == 0.0

def test_all_metrics():
    """Verifies all_metrics combined compiler output keys."""
    pred = [1, 2, 3]
    rel = {2, 4}
    item_cats = {1: 10, 2: 10, 3: 20}
    
    res = Metrics.all_metrics(pred, rel, k=2, item_cats=item_cats)
    assert "precision@2" in res
    assert "recall@2" in res
    assert "ndcg@2" in res
    assert "hit_rate@2" in res
    assert "diversity" in res
    
    assert res["precision@2"] == 0.5
    assert res["hit_rate@2"] == 1.0
