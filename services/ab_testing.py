"""
A/B Experimentation and Cohort Statistical Significance Service.

Implements:
1. Deterministic User Partitioning using MD5 hashing of user:experiment tuples.
2. High-fidelity metrics aggregation (CTR, completion rates, like rates).
3. Two-proportion hypothesis Z-testing via standard mathematical error functions (erf).
4. Direct support for both control vs. treatment evaluations and legacy ABTestingEngine cohorts.
"""

import hashlib
import time
import numpy as np
import math
import scipy.stats as stats
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import logging
logger = logging.getLogger(__name__)

# Inline fallback settings (replaces stale config.settings import)
class _Settings:
    ab_min_samples: int = 100
    ab_significance_level: float = 0.05
    random_seed: int = 42
settings = _Settings()

# ==========================================
# 🚀 Modern A/B Testing Service Components
# ==========================================

class Status(str, Enum):
    DRAFT = "DRAFT"
    RUNNING = "RUNNING"
    DONE = "DONE"

@dataclass
class Group:
    name: str
    fraction: float
    impressions: int = 0
    clicks: int = 0
    watch_completions: int = 0
    likes: int = 0
    dislikes: int = 0
    total_watch_time: float = 0.0

    @property
    def ctr(self) -> float:
        """Click-through rate of this group."""
        return self.clicks / max(self.impressions, 1)

    @property
    def completion_rate(self) -> float:
        """Video watch completion rate of this group."""
        return self.watch_completions / max(self.impressions, 1)

    @property
    def like_rate(self) -> float:
        """Like rate of this group."""
        return self.likes / max(self.impressions, 1)

@dataclass
class Experiment:
    exp_id: str
    description: str
    groups: Dict[str, Group]
    status: Status = Status.DRAFT
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    min_samples: int = 100
    alpha: float = 0.05

class ABTestingService:
    """
    Service coordinating multi-group cohort assignments, conversion tracking, 
    and two-proportion significance tests to drive product decisions.
    """

    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}

    def create(self, exp_id: str, description: str = "", control_frac: float = 0.5, min_samples: int = 100) -> Experiment:
        """Initializes a new Experiment with 'control' and 'treatment' cohorts."""
        groups = {
            "control": Group(name="control", fraction=control_frac),
            "treatment": Group(name="treatment", fraction=1.0 - control_frac)
        }
        exp = Experiment(
            exp_id=exp_id,
            description=description,
            groups=groups,
            min_samples=min_samples,
            alpha=0.05
        )
        self.experiments[exp_id] = exp
        logger.info(f"Created A/B experiment: {exp_id} ({description})")
        return exp

    def start(self, exp_id: str):
        """Transitions experiment status to RUNNING to enable routing assignments."""
        if exp_id in self.experiments:
            exp = self.experiments[exp_id]
            exp.status = Status.RUNNING
            exp.started_at = time.time()
            logger.info(f"Started A/B experiment: {exp_id}")

    def stop(self, exp_id: str):
        """Transitions experiment status to DONE to freeze routing assignments."""
        if exp_id in self.experiments:
            exp = self.experiments[exp_id]
            exp.status = Status.DONE
            exp.ended_at = time.time()
            logger.info(f"Stopped A/B experiment: {exp_id}")

    def assign(self, user_id: Any, exp_id: str) -> Optional[str]:
        """
        Deterministically routes a User ID to control or treatment group using MD5 hashing.
        Returns None if the experiment is not active (not RUNNING).
        """
        if exp_id not in self.experiments:
            return None
        
        exp = self.experiments[exp_id]
        if exp.status != Status.RUNNING:
            return None

        # MD5 reproducible, well-distributed partitioning key
        hasher = hashlib.md5(f"{user_id}:{exp_id}".encode("utf-8"))
        hash_val = int(hasher.hexdigest(), 16)
        fraction = (hash_val % 10000) / 10000.0

        control_group = exp.groups.get("control")
        if control_group is None:
            return "treatment"

        if fraction < control_group.fraction:
            return "control"
        else:
            return "treatment"

    def record(self, exp_id: str, group: str, event: str, watch_time: float = 0.0):
        """Aggregates user conversion behavior events into the target cohort group."""
        if exp_id not in self.experiments:
            return
        
        exp = self.experiments[exp_id]
        if exp.status != Status.RUNNING:
            return

        if group not in exp.groups:
            return

        g = exp.groups[group]
        
        if event == "impression":
            g.impressions += 1
        elif event == "click":
            g.clicks += 1
        elif event == "watch_complete":
            g.watch_completions += 1
        elif event == "like":
            g.likes += 1
        elif event == "dislike":
            g.dislikes += 1
        
        if watch_time > 0:
            g.total_watch_time += watch_time

    def analyze(self, exp_id: str) -> dict:
        """
        Runs mathematical hypothesis tests comparing Control vs. Treatment metrics
        to compile statistical significances and relative lifts.
        """
        if exp_id not in self.experiments:
            return {"error": "Experiment not found."}

        exp = self.experiments[exp_id]
        control = exp.groups["control"]
        treatment = exp.groups["treatment"]

        report = {
            "exp_id": exp.exp_id,
            "status": exp.status,
            "control": {
                "impressions": control.impressions,
                "ctr": control.ctr,
                "completion_rate": control.completion_rate,
                "like_rate": control.like_rate
            },
            "treatment": {
                "impressions": treatment.impressions,
                "ctr": treatment.ctr,
                "completion_rate": treatment.completion_rate,
                "like_rate": treatment.like_rate
            }
        }

        # Check if both groups satisfy minimum sample limits
        if control.impressions >= exp.min_samples and treatment.impressions >= exp.min_samples:
            report["ctr_test"] = self._ztest(
                control.clicks, control.impressions,
                treatment.clicks, treatment.impressions,
                exp.alpha
            )
            report["completion_test"] = self._ztest(
                control.watch_completions, control.impressions,
                treatment.watch_completions, treatment.impressions,
                exp.alpha
            )
        else:
            report["message"] = f"Insufficient samples to compute Z-Test (Control: {control.impressions}/{exp.min_samples}, Treatment: {treatment.impressions}/{exp.min_samples})."

        return report

    @staticmethod
    def _ztest(sA: int, nA: int, sB: int, nB: int, alpha: float) -> dict:
        """
        Performs a two-proportion Z-test. Computes pooled conversion rates,
        standard error estimates, Z-scores, and exact two-tailed P-values via math.erf.
        """
        pA = sA / max(nA, 1)
        pB = sB / max(nB, 1)
        
        pooled = (sA + sB) / max(nA + nB, 1)
        se = math.sqrt(pooled * (1.0 - pooled) * (1.0/max(nA, 1) + 1.0/max(nB, 1)))
        
        z = (pB - pA) / max(se, 1e-10)
        
        # Two-tailed P-value calculation using math.erf:
        # P = 2 * (1 - cdf(|Z|)) where cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
        pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        
        lift = (pB - pA) / max(pA, 1e-10)
        significant = pval < alpha

        if significant:
            verdict = "SHIP treatment" if z > 0 else "KEEP control"
        else:
            verdict = "CONTINUE"

        return {
            "z_stat": float(z),
            "p_value": float(pval),
            "significant": bool(significant),
            "relative_lift": float(lift),
            "control_rate": float(pA),
            "treatment_rate": float(pB),
            "verdict": verdict
        }

    def simulate(self, exp_id: str, n_users: int = 3000, ctrl_ctr: float = 0.05, treat_ctr: float = 0.06,
                 ctrl_completion: float = 0.15, treat_completion: float = 0.18) -> dict:
        """Simulates conversion telemetry across partitioned cohorts and generates an analytical report."""
        # 1. Create and start the experiment
        self.create(exp_id=exp_id, description=f"Simulation {exp_id}", control_frac=0.5, min_samples=100)
        self.start(exp_id)

        np.random.seed(42)

        # 2. Assign and record conversion events for n_users
        for uid in range(n_users):
            group = self.assign(uid, exp_id)
            if not group:
                continue

            self.record(exp_id, group, "impression")
            
            # Select probabilities based on group routing
            ctr = ctrl_ctr if group == "control" else treat_ctr
            completion = ctrl_completion if group == "control" else treat_completion
            like_prob = 0.10 if group == "control" else 0.12

            # Click conversion check
            if np.random.random() < ctr:
                self.record(exp_id, group, "click")
                
                # Watch completion check
                if np.random.random() < completion:
                    self.record(exp_id, group, "watch_complete")
                
                # Like/Dislike checks
                if np.random.random() < like_prob:
                    self.record(exp_id, group, "like")
                elif np.random.random() < 0.02:
                    self.record(exp_id, group, "dislike")

        # 3. Stop and analyze
        self.stop(exp_id)
        return self.analyze(exp_id)


# ==========================================
# 🔄 Backward Compatibility Legacy Engine
# ==========================================

class ABTestingEngine:
    """
    Legacy class supporting deterministic hashing across Control vs Treatment A/B cohorts.
    Ensures that existing simulator scripts and dashboard interfaces continue to work.
    """
    def __init__(self):
        self.min_samples = settings.ab_min_samples
        self.alpha = settings.ab_significance_level
        self.buckets = ["CONTROL", "TREATMENT_A", "TREATMENT_B"]

    def assign_user(self, user_id: int) -> str:
        """Deterministically hashes User IDs into one of 3 experimental buckets."""
        hasher = hashlib.md5(str(user_id).encode("utf-8"))
        hash_val = int(hasher.hexdigest(), 16)
        remainder = hash_val % 100
        if remainder < 33:
            return "CONTROL"
        elif remainder < 66:
            return "TREATMENT_A"
        else:
            return "TREATMENT_B"

    def calculate_z_test(self, clicks_1: int, n_1: int, clicks_2: int, n_2: int) -> dict:
        """Calculates a two-proportion Z-test to evaluate statistical significance."""
        if n_1 < self.min_samples or n_2 < self.min_samples:
            return {
                "z_score": 0.0,
                "p_value": 1.0,
                "significant": False,
                "recommendation": "Collect more samples (under min threshold)."
            }
            
        p_1 = clicks_1 / n_1
        p_2 = clicks_2 / n_2
        
        p_pooled = (clicks_1 + clicks_2) / (n_1 + n_2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_1 + 1/n_2))
        
        if se == 0.0:
            return {
                "z_score": 0.0,
                "p_value": 1.0,
                "significant": False,
                "recommendation": "No variance in outcomes."
            }
            
        z_score = (p_2 - p_1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) # Two-tailed test
        significant = p_value < self.alpha
        
        if significant:
            rec = "Deploy Treatment (highly significant uplift)! 🚀" if z_score > 0 else "Retain Control (Treatment performed worse)."
        else:
            rec = "Retain Control (uplift not statistically significant)."
            
        return {
            "z_score": float(z_score),
            "p_value": float(p_value),
            "significant": bool(significant),
            "recommendation": rec
        }

    def simulate_experiment(self, num_cohort_users: int = 5000) -> dict:
        """Simulates metrics (CTR, Watch Rate) across partitioned cohorts."""
        results = {
            "CONTROL": {"users": 0, "clicks": 0, "likes": 0, "watch_hours": 0.0},
            "TREATMENT_A": {"users": 0, "clicks": 0, "likes": 0, "watch_hours": 0.0},
            "TREATMENT_B": {"users": 0, "clicks": 0, "likes": 0, "watch_hours": 0.0}
        }
        
        np.random.seed(settings.random_seed)
        
        probs = {
            "CONTROL": {"click": 0.12, "like": 0.10, "mean_watch": 0.3},
            "TREATMENT_A": {"click": 0.22, "like": 0.18, "mean_watch": 0.5},
            "TREATMENT_B": {"click": 0.29, "like": 0.25, "mean_watch": 0.6}
        }
        
        for uid in range(num_cohort_users):
            bucket = self.assign_user(uid)
            results[bucket]["users"] += 1
            
            p = probs[bucket]
            clicked = 1 if np.random.random() < p["click"] else 0
            if clicked == 1:
                results[bucket]["clicks"] += 1
                watch_ratio = np.clip(np.random.normal(p["mean_watch"], 0.15), 0.05, 1.0)
                results[bucket]["watch_hours"] += float(watch_ratio * 0.1)
                
                if np.random.random() < p["like"]:
                    results[bucket]["likes"] += 1
                    
        return results
