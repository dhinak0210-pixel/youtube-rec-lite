"""
Recommendation System Offline Evaluation Metrics.

Implements standard Information Retrieval (IR) metrics used to evaluate the accuracy,
coverage, and diversity of recommendation systems. Includes:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate
- Catalog Coverage
- Category Diversity
"""

import numpy as np
from typing import List, Set, Dict, Optional

class Metrics:
    """
    Standard Information Retrieval (IR) and recommendation evaluation metrics.
    
    All methods are implemented as static methods targeting exact mathematical formulations
    for offline candidate quality ranking assessments.
    """

    @staticmethod
    def precision_at_k(rec: List[int], rel: Set[int], k: int = 10) -> float:
        """
        Calculates the fraction of top-k recommendations that are relevant.

        Formula:
        Precision@K = (Relevant Recommended Items in Top K) / K

        Example:
        >>> Metrics.precision_at_k([1, 2, 3, 4], {2, 4}, k=3)
        0.3333333333333333
        """
        if k <= 0 or not rec:
            return 0.0
        rec_k = rec[:k]
        hits_in_top_k = sum(1 for x in rec_k if x in rel)
        return hits_in_top_k / k

    @staticmethod
    def recall_at_k(rec: List[int], rel: Set[int], k: int = 10) -> float:
        """
        Calculates the fraction of relevant items that appear in the top-k recommendations.

        Formula:
        Recall@K = (Relevant Recommended Items in Top K) / (Total Relevant Items)

        Example:
        >>> Metrics.recall_at_k([1, 2, 3, 4], {2, 4, 5}, k=3)
        0.3333333333333333
        """
        if not rel or not rec or k <= 0:
            return 0.0
        rec_k = rec[:k]
        hits_in_top_k = sum(1 for x in rec_k if x in rel)
        return hits_in_top_k / len(rel)

    @staticmethod
    def ndcg_at_k(rec: List[int], rel: Set[int], k: int = 10) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG@K) of top-k recommendations.
        Weights higher positions with logarithmic discounts.

        Formula:
        DCG@K = Sum_{i=1}^K [ I(rec[i] in rel) / log2(i + 1) ]
        IDCG@K = Sum_{i=1}^{min(|rel|, K)} [ 1 / log2(i + 1) ]
        NDCG@K = DCG@K / IDCG@K

        Example:
        >>> Metrics.ndcg_at_k([1, 2, 3], {2, 4}, k=2)
        0.38685280723454163
        """
        if not rel or not rec or k <= 0:
            return 0.0
        
        rec_k = rec[:k]
        
        # Calculate Discounted Cumulative Gain (DCG)
        dcg = 0.0
        for idx, item in enumerate(rec_k):
            if item in rel:
                dcg += 1.0 / np.log2(idx + 2)
                
        # Calculate Ideal Discounted Cumulative Gain (IDCG)
        idcg = 0.0
        ideal_hits = min(len(rel), k)
        for idx in range(ideal_hits):
            idcg += 1.0 / np.log2(idx + 2)
            
        return dcg / max(idcg, 1e-10)

    @staticmethod
    def hit_rate(rec: List[int], rel: Set[int], k: int = 10) -> float:
        """
        Calculates the Hit Rate@K, which is 1.0 if at least one relevant item appears
        in the top-k recommendations, else 0.0.

        Example:
        >>> Metrics.hit_rate([1, 2, 3], {2, 4}, k=2)
        1.0
        """
        if not rel or not rec or k <= 0:
            return 0.0
        
        rec_k = rec[:k]
        for item in rec_k:
            if item in rel:
                return 1.0
        return 0.0

    @staticmethod
    def coverage(all_recs: List[List[int]], total_items: int) -> float:
        """
        Calculates the fraction of the catalog that appears in any recommendation list.

        Formula:
        Coverage = |Unique Recommended Items across all lists| / (Total Catalog Items)

        Example:
        >>> Metrics.coverage([[1, 2], [2, 3]], total_items=5)
        0.6
        """
        if total_items <= 0:
            return 0.0
        
        unique_recommended_items = set()
        for rec in all_recs:
            unique_recommended_items.update(rec)
            
        return len(unique_recommended_items) / total_items

    @staticmethod
    def diversity(rec: List[int], item_cats: Dict[int, int]) -> float:
        """
        Calculates the fraction of unique categories present in the recommendation list.

        Formula:
        Diversity = |Unique Categories represented by recommended items| / |Recommendation list|

        Example:
        >>> Metrics.diversity([1, 2, 3], {1: 10, 2: 10, 3: 20})
        0.6666666666666666
        """
        if not rec:
            return 0.0
        
        unique_categories = set()
        for item in rec:
            if item in item_cats:
                unique_categories.add(item_cats[item])
                
        return len(unique_categories) / max(len(rec), 1)

    @staticmethod
    def all_metrics(rec: List[int], rel: Set[int], k: int = 10, item_cats: Optional[Dict[int, int]] = None) -> Dict[str, float]:
        """
        Computes all standard precision, recall, hit rate, and NDCG metrics in one call.
        Adds catalog diversity if category dictionary map is provided.

        Example:
        >>> Metrics.all_metrics([1, 2, 3], {2, 4}, k=2)
        {'precision@2': 0.5, 'recall@2': 0.5, 'ndcg@2': 0.38685280723454163, 'hit_rate@2': 1.0}
        """
        res = {
            f"precision@{k}": Metrics.precision_at_k(rec, rel, k),
            f"recall@{k}": Metrics.recall_at_k(rec, rel, k),
            f"ndcg@{k}": Metrics.ndcg_at_k(rec, rel, k),
            f"hit_rate@{k}": Metrics.hit_rate(rec, rel, k)
        }
        if item_cats is not None:
            res["diversity"] = Metrics.diversity(rec, item_cats)
        return res


# ==========================================
# 🔄 Backward Compatibility Legacy Engine
# ==========================================

class RecommendationMetrics:
    """
    Legacy recommendation metrics evaluator class supporting evaluate_all loops.
    """
    @staticmethod
    def precision_at_k(actual: list, predicted: list, k: int) -> float:
        if not predicted or k <= 0:
            return 0.0
        pred_k = predicted[:k]
        actual_set = set(actual)
        relevant = [item for item in pred_k if item in actual_set]
        return len(relevant) / k

    @staticmethod
    def recall_at_k(actual: list, predicted: list, k: int) -> float:
        if not actual or not predicted or k <= 0:
            return 0.0
        pred_k = predicted[:k]
        actual_set = set(actual)
        relevant = [item for item in pred_k if item in actual_set]
        return len(relevant) / len(actual)

    @staticmethod
    def hit_rate_at_k(actual: list, predicted: list, k: int) -> float:
        if not actual or not predicted or k <= 0:
            return 0.0
        pred_k = set(predicted[:k])
        for item in actual:
            if item in pred_k:
                return 1.0
        return 0.0

    @staticmethod
    def ndcg_at_k(actual: list, predicted: list, k: int) -> float:
        if not actual or not predicted or k <= 0:
            return 0.0
        pred_k = predicted[:k]
        actual_set = set(actual)
        dcg = 0.0
        for idx, item in enumerate(pred_k):
            if item in actual_set:
                dcg += 1.0 / np.log2(idx + 2)
        idcg = 0.0
        ideal_hits = min(len(actual), k)
        for idx in range(ideal_hits):
            idcg += 1.0 / np.log2(idx + 2)
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @classmethod
    def evaluate_all(cls, actual: list, predicted: list, k: int = 10) -> dict:
        return {
            f"precision_at_{k}": cls.precision_at_k(actual, predicted, k),
            f"recall_at_{k}": cls.recall_at_k(actual, predicted, k),
            f"hit_rate_at_{k}": cls.hit_rate_at_k(actual, predicted, k),
            f"ndcg_at_{k}": cls.ndcg_at_k(actual, predicted, k)
        }
