import numpy as np
from typing import Dict, List
from src.utils.logger import logger

class ABMetricsEvaluator:
    """
    Computes online tracking statistics for A/B experiment groups.
    Calculates Click-Through Rate (CTR), Average Watch Ratios, and NDCG values.
    """
    def __init__(self):
        # Accumulators: { group: [values] }
        self.impressions = {"Control": 0, "Treatment": 0}
        self.clicks = {"Control": 0, "Treatment": 0}
        self.watch_ratios = {"Control": [], "Treatment": []}
        self.likes = {"Control": 0, "Treatment": 0}

    def log_impression(self, group: str):
        """
        Increments impression count for the target group.
        """
        if group in self.impressions:
            self.impressions[group] += 1

    def log_interaction(self, group: str, click: int, watch_ratio: float, like: int):
        """
        Records a real-time interaction for metric logging.
        """
        if group not in self.impressions:
            return
            
        if click:
            self.clicks[group] += 1
            self.watch_ratios[group].append(watch_ratio)
            if like:
                self.likes[group] += 1
                
        logger.debug(f"Interaction logged for group {group} - Click: {click}, WatchRatio: {watch_ratio:.2f}, Like: {like}")

    def calculate_ndcg(self, relevance_scores: List[float], k: int = 10) -> float:
        """
        Helper to calculate NDCG (Normalized Discounted Cumulative Gain) for recommendations.
        """
        if not relevance_scores:
            return 0.0
            
        relevance_scores = relevance_scores[:k]
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])
        
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
        
        if idcg == 0.0:
            return 0.0
            
        return float(dcg / idcg)

    def get_group_metrics(self, group: str) -> Dict[str, float]:
        """
        Summarizes performance scores for a cohort.
        """
        impr = self.impressions.get(group, 0)
        clks = self.clicks.get(group, 0)
        w_ratios = self.watch_ratios.get(group, [])
        lks = self.likes.get(group, 0)
        
        ctr = (clks / impr) if impr > 0 else 0.0
        avg_watch = np.mean(w_ratios) if len(w_ratios) > 0 else 0.0
        like_ratio = (lks / clks) if clks > 0 else 0.0
        
        return {
            "impressions": impr,
            "clicks": clks,
            "ctr": float(ctr),
            "avg_watch_ratio": float(avg_watch),
            "like_ratio": float(like_ratio)
        }

    def get_report(self) -> Dict[str, Dict[str, float]]:
        """
        Produces comparative summary report of all experimental groups.
        """
        return {
            "Control": self.get_group_metrics("Control"),
            "Treatment": self.get_group_metrics("Treatment")
        }
