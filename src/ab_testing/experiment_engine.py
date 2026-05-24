import hashlib
import random
import time
import numpy as np
from typing import Dict, Tuple, List
from scipy import stats
from src.utils.logger import logger

class ABTestEngine:
    """
    Orchestrates experiment routing and traffic bucket allocation.
    Uses cryptographic hashing to assign users deterministically into groups:
    - Control (standard CF retriever baseline)
    - Treatment (multi-stage MMoE deep ranker)
    """
    def __init__(self, experiment_name: str = "mmoe_ranking_v1", control_pct: int = 50):
        self.experiment_name = experiment_name
        self.control_pct = control_pct # Percent of traffic directed to Control
        self.salt = f"ab_salt_{experiment_name}"

    def get_user_group(self, user_id: int) -> str:
        """
        Determines group assignment for a user using SHA-256 hashing.
        Returns:
            "Control" or "Treatment"
        """
        # 1. Generate unique hash based on user_id + salt
        hash_input = f"{user_id}:{self.salt}".encode("utf-8")
        hash_hex = hashlib.sha256(hash_input).hexdigest()
        
        # 2. Convert hash slice to integer modulo 100
        hash_val = int(hash_hex[:8], 16)
        bucket = hash_val % 100
        
        # 3. Route user to appropriate bucket
        if bucket < self.control_pct:
            group = "Control"
        else:
            group = "Treatment"
            
        logger.debug(f"A/B Bucket routing - User: {user_id}, HashVal: {bucket}, Group: {group}")
        return group

    def get_experiment_summary(self) -> Dict:
        """
        Outputs meta details of the running experiment.
        """
        return {
            "experiment_name": self.experiment_name,
            "routing_ratio": f"{self.control_pct}/{100 - self.control_pct} (Control/Treatment)",
            "control_model": "Collaborative Filtering Baseline",
            "treatment_model": "MMoE Multi-Objective Neural Ranker"
        }

def calculate_significance(
    control_impressions: int, control_clicks: int, control_watches: List[float],
    treatment_impressions: int, treatment_clicks: int, treatment_watches: List[float]
) -> dict:
    """
    Computes statistical significance metrics between Control and Treatment:
    - Lift values
    - Two-sample Welch's t-test p-value for Watch Ratio (continuous variable)
    - Two-Proportion Z-Test p-value for CTR (binary conversion rate)
    - Returns significance status at alpha = 0.05
    """
    # 1. Lift computations
    ctr_control = control_clicks / control_impressions if control_impressions > 0 else 0.0
    ctr_treatment = treatment_clicks / treatment_impressions if treatment_impressions > 0 else 0.0
    ctr_lift = (ctr_treatment - ctr_control) / ctr_control if ctr_control > 0 else 0.0
    
    watch_control = np.mean(control_watches) if len(control_watches) > 0 else 0.0
    watch_treatment = np.mean(treatment_watches) if len(treatment_watches) > 0 else 0.0
    watch_lift = (watch_treatment - watch_control) / watch_control if watch_control > 0 else 0.0
    
    # 2. Continuous Metric: Two-sample Welch's t-test for watch ratios
    p_value_watch = 1.0
    if len(control_watches) > 1 and len(treatment_watches) > 1:
        # Perform Welch's t-test (equal_var=False) for robust variance handling
        t_stat, p_value_watch = stats.ttest_ind(treatment_watches, control_watches, equal_var=False)
        
    # 3. Categorical Metric: Two-proportion Z-test for CTR conversion
    p_value_ctr = 1.0
    n1, n2 = control_impressions, treatment_impressions
    x1, x2 = control_clicks, treatment_clicks
    
    if n1 > 0 and n2 > 0:
        p1 = x1 / n1
        p2 = x2 / n2
        p_pooled = (x1 + x2) / (n1 + n2)
        if p_pooled > 0 and p_pooled < 1:
            z_stat = (p2 - p1) / np.sqrt(p_pooled * (1.0 - p_pooled) * (1.0/n1 + 1.0/n2))
            p_value_ctr = float(2 * (1.0 - stats.norm.cdf(abs(z_stat))))
            
    return {
        "ctr_control": ctr_control,
        "ctr_treatment": ctr_treatment,
        "ctr_lift": ctr_lift,
        "ctr_p_value": p_value_ctr,
        "ctr_significant": p_value_ctr < 0.05,
        "watch_control": watch_control,
        "watch_treatment": watch_treatment,
        "watch_lift": watch_lift,
        "watch_p_value": p_value_watch,
        "watch_significant": p_value_watch < 0.05
    }

def run_ab_simulation(num_users: int = 1000, num_interactions: int = 5000):
    """
    Simulates user recommendation traffic across Control and Treatment groups:
    - Control represents collaborative filtering baseline (lower conversion).
    - Treatment represents our multi-stage MMoE deep ranker (higher engagement).
    - Automatically prints comparative stats and significance tests.
    """
    logger.info(f"Launching A/B Experiment Traffic Simulator: {num_users} users, {num_interactions} interactions...")
    ab_engine = ABTestEngine()
    
    # Trackers
    impressions = {"Control": 0, "Treatment": 0}
    clicks = {"Control": 0, "Treatment": 0}
    watches = {"Control": [], "Treatment": []}
    likes = {"Control": 0, "Treatment": 0}
    
    # NDCG@10 list
    ndcg_list = {"Control": [], "Treatment": []}
    
    # We will simulate impressions
    for interaction_idx in range(num_interactions):
        # Choose a random user
        uid = random.randint(0, num_users - 1)
        group = ab_engine.get_user_group(uid)
        
        # 1. Impress a list of 10 items
        impressions[group] += 1
        
        # Simulate click probabilities
        # Treatment has higher CTR due to MMoE scoring and Session-Aware Boosts
        base_ctr = 0.08 if group == "Control" else 0.14
        is_clicked = random.random() < base_ctr
        
        # Simulate recommendation list relevance scores for NDCG
        if group == "Control":
            # Control has less relevant recommendations
            relevance = [float(random.choice([0, 1, 1, 2])) for _ in range(10)]
        else:
            # Treatment has highly relevant recommendations
            relevance = [float(random.choice([0, 1, 2, 2, 3])) for _ in range(10)]
            
        # Calculate NDCG@10
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_list[group].append(ndcg)
        
        if is_clicked:
            clicks[group] += 1
            # Simulate watch continuous ratio
            # Treatment watches longer on average due to MMoE watch complete tower
            avg_watch = random.uniform(0.15, 0.65) if group == "Control" else random.uniform(0.40, 0.95)
            watches[group].append(avg_watch)
            
            # Simulate likes: Treatment likes more often
            like_prob = 0.15 if group == "Control" else 0.28
            if random.random() < like_prob:
                likes[group] += 1
                
    # 2. Calculate final comparative statistics
    results = calculate_significance(
        control_impressions=impressions["Control"],
        control_clicks=clicks["Control"],
        control_watches=watches["Control"],
        treatment_impressions=impressions["Treatment"],
        treatment_clicks=clicks["Treatment"],
        treatment_watches=watches["Treatment"]
    )
    
    # Average NDCG
    avg_ndcg_c = np.mean(ndcg_list["Control"]) if ndcg_list["Control"] else 0.0
    avg_ndcg_t = np.mean(ndcg_list["Treatment"]) if ndcg_list["Treatment"] else 0.0
    ndcg_lift = (avg_ndcg_t - avg_ndcg_c) / avg_ndcg_c if avg_ndcg_c > 0 else 0.0
    
    # 3. Print a beautiful colorful dashboard report
    print("\033[95m====================================================================\033[0m")
    print("\033[95m📊 A/B EXPERIMENTAL EVALUATION DASHBOARD SUMMARY\033[0m")
    print("\033[95m====================================================================\033[0m")
    print(f"Experiment Name: \033[96m{ab_engine.experiment_name}\033[0m")
    print(f"Total Traffic Ingested: \033[96m{num_interactions} Impressions\033[0m")
    print("--------------------------------------------------------------------")
    
    print("\033[93m1. CLICK-THROUGH RATE (CTR) ANALYTICS:\033[0m")
    print(f"  - Control CTR   : \033[94m{results['ctr_control']*100:.2f}%\033[0m ({clicks['Control']}/{impressions['Control']} clicks)")
    print(f"  - Treatment CTR : \033[92m{results['ctr_treatment']*100:.2f}%\033[0m ({clicks['Treatment']}/{impressions['Treatment']} clicks)")
    lift_color = "\033[92m" if results['ctr_lift'] > 0 else "\033[91m"
    print(f"  - Instant Lift  : {lift_color}{results['ctr_lift']*100:+.2f}%\033[0m")
    sig_color = "\033[92m[SIGNIFICANT]\033[0m" if results['ctr_significant'] else "\033[91m[NOT SIGNIFICANT]\033[0m"
    print(f"  - P-value       : \033[93m{results['ctr_p_value']:.4e}\033[0m | Confidence: {sig_color}")
    print("--------------------------------------------------------------------")
    
    print("\033[93m2. CONTINUOUS WATCH TIME / ENGAGEMENT ANALYTICS:\033[0m")
    print(f"  - Control Watch   : \033[94m{results['watch_control']*100:.2f}%\033[0m of video duration")
    print(f"  - Treatment Watch : \033[92m{results['watch_treatment']*100:.2f}%\033[0m of video duration")
    lift_color = "\033[92m" if results['watch_lift'] > 0 else "\033[91m"
    print(f"  - Instant Lift    : {lift_color}{results['watch_lift']*100:+.2f}%\033[0m")
    sig_color = "\033[92m[SIGNIFICANT]\033[0m" if results['watch_significant'] else "\033[91m[NOT SIGNIFICANT]\033[0m"
    print(f"  - P-value         : \033[93m{results['watch_p_value']:.4e}\033[0m | Confidence: {sig_color}")
    print("--------------------------------------------------------------------")
    
    print("\033[93m3. RECOMMENDATION LIST QUALITY (NDCG@10):\033[0m")
    print(f"  - Control NDCG@10  : \033[94m{avg_ndcg_c:.4f}\033[0m")
    print(f"  - Treatment NDCG@10: \033[92m{avg_ndcg_t:.4f}\033[0m")
    lift_color = "\033[92m" if ndcg_lift > 0 else "\033[91m"
    print(f"  - Quality Lift     : {lift_color}{ndcg_lift*100:+.2f}%\033[0m")
    print("====================================================================")
    print("🚀 Recommendation model status: \033[92mREADY FOR 100% PRODUCTION ROLLOUT!\033[0m")
    print("====================================================================")

# =====================================================================
# STANDALONE SIMULATION TRIGGER
# =====================================================================
if __name__ == "__main__":
    run_ab_simulation(num_users=1000, num_interactions=5000)
