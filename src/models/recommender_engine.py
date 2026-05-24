import time
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from src.utils.logger import logger
from src.api.schemas import RecommendationResponse, VideoRecommendation
from src.models.mmoe_ranking import MultiObjectiveRankingModel

class RecommendationEngine:
    """
    Unified Orchestrator combining:
    1. Collaborative Filtering (behavioral similarity)
    2. ALS Matrix Factorization (latent factors)
    3. BERT4Rec (sequential transformer)
    4. SocialGraphRecommender (GraphSAGE / friend watched counts)
    5. Multi-Objective MMoE (final neural deep ranking)
    6. Cold Start Handler (fallback popular retrieval)
    7. Real-Time Streaming context (trending & user session moods)
    """
    def __init__(self, config: Dict):
        self.cf_model = config.get("cf_model")
        self.als_model = config.get("als_model")
        self.bert_model = config.get("bert_model")
        self.social_graph = config.get("social_graph")
        self.mmoe_model = config.get("mmoe_model") # Can be MMoERankingEngine or MultiObjectiveRankingModel
        self.cold_start_handler = config.get("cold_start_handler")
        self.stream_processor = config.get("stream_processor")
        self.preprocessor = config.get("preprocessor")
        self.users_df = config.get("users_df")
        self.videos_df = config.get("videos_df")
        self.ab_engine = config.get("ab_engine")

        # Source candidate selection weights
        self.source_weights = config.get("source_weights", {
            "cf": 1.0,
            "mf": 1.0,
            "bert": 1.0,
            "social": 1.2,
            "trending": 1.1
        })

        # Latency statistics tracking (in milliseconds)
        self.latency_metrics = {
            "candidate_cf": [],
            "candidate_mf": [],
            "candidate_bert": [],
            "candidate_social": [],
            "candidate_trending": [],
            "ranking": [],
            "total": []
        }
        self.ranking_latencies = []
        self.total_latencies = []
        self.cache_hits = 0
        self.total_queries = 0


    def generate_candidates(self, user_id: int, context: Optional[Dict] = None, n: int = 500) -> Tuple[List[int], Dict[int, List[str]], Dict[str, int]]:
        """
        Retrieves candidates from diverse model sources and merges them:
        - CF (n=150)
        - MF (n=150)
        - BERT4Rec (n=100)
        - Social (n=50)
        - Trending (n=50)
        """
        candidate_sources = {} # item_id -> list of sources it was fetched from
        sources_breakdown = {"cf": 0, "mf": 0, "bert": 0, "social": 0, "trending": 0}

        # 1. Collaborative Filtering (behavioral similarity)
        t0 = time.perf_counter()
        cf_items = []
        if self.cf_model:
            try:
                cf_recs = self.cf_model.retrieve_candidates(user_id, top_n=150)
                cf_items = [item for item, _ in cf_recs]
            except Exception as e:
                logger.debug(f"CF candidate retrieval failed: {e}")
        self.latency_metrics["candidate_cf"].append((time.perf_counter() - t0) * 1000)

        # 2. Matrix Factorization (latent space factors)
        t0 = time.perf_counter()
        mf_items = []
        if self.als_model:
            try:
                mf_recs = self.als_model.retrieve_candidates(user_id, top_n=150)
                mf_items = [item for item, _ in mf_recs]
            except Exception as e:
                logger.debug(f"MF candidate retrieval failed: {e}")
        self.latency_metrics["candidate_mf"].append((time.perf_counter() - t0) * 1000)

        # 3. BERT4Rec (Sequential Interest Transformer)
        t0 = time.perf_counter()
        bert_items = []
        if self.bert_model:
            try:
                # Retrieve user session history from streaming Flink window
                history = []
                if self.stream_processor:
                    history = self.stream_processor.get_user_session(user_id)
                if not history:
                    # Fallback to random recent clicks
                    history = [random.randint(1, 100) for _ in range(5)]
                bert_recs = self.bert_model.predict_next_items(history)
                bert_items = [item for item, _ in bert_recs[:100]]
            except Exception as e:
                logger.debug(f"BERT4Rec candidate retrieval failed: {e}")
        self.latency_metrics["candidate_bert"].append((time.perf_counter() - t0) * 1000)

        # 4. Social Circles (friend watches)
        t0 = time.perf_counter()
        social_items = []
        if self.social_graph:
            try:
                friend_watches = self.social_graph.get_friend_watched_items(user_id)
                sorted_friends = sorted(friend_watches.items(), key=lambda x: x[1], reverse=True)[:50]
                social_items = [item for item, _ in sorted_friends]
            except Exception as e:
                logger.debug(f"Social candidate retrieval failed: {e}")
        self.latency_metrics["candidate_social"].append((time.perf_counter() - t0) * 1000)

        # 5. Flink Real-Time Trending (explore popularity)
        t0 = time.perf_counter()
        trending_items = []
        if self.stream_processor:
            try:
                trending_recs = self.stream_processor.get_trending_items(top_k=50)
                trending_items = [item for item, _ in trending_recs]
            except Exception as e:
                logger.debug(f"Trending candidate retrieval failed: {e}")
        self.latency_metrics["candidate_trending"].append((time.perf_counter() - t0) * 1000)

        # Ingestion merge & source logging
        for items, source_name in [(cf_items, "cf"), (mf_items, "mf"), (bert_items, "bert"), (social_items, "social"), (trending_items, "trending")]:
            for item in items:
                if item not in candidate_sources:
                    candidate_sources[item] = []
                candidate_sources[item].append(source_name)
                sources_breakdown[source_name] += 1

        merged_candidates = list(candidate_sources.keys())
        if not merged_candidates and self.videos_df is not None:
            fallback_vids = self.videos_df["video_id"].tolist()
            for item in fallback_vids:
                if item not in candidate_sources:
                    candidate_sources[item] = ["trending"]
                sources_breakdown["trending"] += 1
            merged_candidates = fallback_vids

        random.shuffle(merged_candidates) # Shuffle before truncation to ensure diversity
        final_pool = merged_candidates[:n]


        return final_pool, candidate_sources, sources_breakdown

    def rank_candidates(self, user_id: int, candidates: List[int], context: Optional[Dict] = None) -> List[Tuple[int, float, Dict[str, float]]]:
        """
        Ranks candidates using Multi-Objective Deep MMoE Network.
        Adds social, trending, and user mood boosts, and applies a diversity penalty.
        """
        if not candidates:
            return []

        t0 = time.perf_counter()
        scored_candidates = []

        # 1. Build Feature Matrix for MMoE candidate model
        # Features: [user_idx, gender_idx, country_idx, age_norm, video_idx, category_idx, duration_norm]
        features = []
        candidate_ids = []
        
        # User details mapping
        try:
            user_row = self.users_df[self.users_df["user_id"] == user_id].iloc[0]
            u_idx = self.preprocessor.user_to_idx.get(user_id, 0)
            u_gen = self.preprocessor.gender_to_idx.get(user_row["gender"], 0)
            u_ctr = self.preprocessor.country_to_idx.get(user_row["country"], 0)
            u_age = user_row["age"] / 100.0
        except Exception:
            # Cold start user default values
            u_idx, u_gen, u_ctr, u_age = 0, 0, 0, 0.3

        # Compile video matrices
        for vid in candidates:
            try:
                v_row = self.videos_df[self.videos_df["video_id"] == vid].iloc[0]
                v_idx = self.preprocessor.video_to_idx.get(vid, 0)
                v_cat = self.preprocessor.cat_to_idx.get(v_row["category"], 0)
                v_dur = v_row["duration"] / self.videos_df["duration"].max()
                features.append([u_idx, u_gen, u_ctr, u_age, v_idx, v_cat, v_dur])
                candidate_ids.append(vid)
            except Exception:
                continue

        if not features:
            return []

        features_np = np.array(features)
        
        # 2. Run MMoE Deep Multi-Objective Model to get Click, Watch, Like, Dislike scores
        pred_dict = {"click": np.zeros(len(features)), "watch_complete": np.zeros(len(features)), "like": np.zeros(len(features)), "dislike": np.zeros(len(features))}
        
        if self.mmoe_model:
            try:
                # If wrapped inside MMoERankingEngine, fetch underlying model
                model_net = getattr(self.mmoe_model, "model", self.mmoe_model)
                model_net.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(features_np, dtype=torch.float32)
                    
                    # Handle both standard four-objective model and legacy two-objective model
                    if isinstance(model_net, MultiObjectiveRankingModel):
                        outputs = model_net(X_tensor)
                        for task in ['click', 'watch_complete', 'like', 'dislike']:
                            pred_dict[task] = outputs[task].numpy()
                    else:
                        # Legacy clicks & watch ratio model fallback
                        pred_click, pred_watch = model_net(X_tensor)
                        pred_dict["click"] = pred_click.numpy()
                        pred_dict["watch_complete"] = pred_watch.numpy()
                        pred_dict["like"] = pred_click.numpy() * 0.3 # Mock liked ratio correlation
                        pred_dict["dislike"] = (1.0 - pred_click.numpy()) * 0.1
            except Exception as e:
                logger.debug(f"MMoE scoring failed: {e}")
                # Mock fallback
                size = len(features)
                pred_dict = {
                    "click": np.random.uniform(0.1, 0.9, size),
                    "watch_complete": np.random.uniform(0.2, 0.9, size),
                    "like": np.random.uniform(0.05, 0.4, size),
                    "dislike": np.random.uniform(0.01, 0.1, size)
                }
        else:
            size = len(features)
            pred_dict = {
                "click": np.random.uniform(0.1, 0.9, size),
                "watch_complete": np.random.uniform(0.2, 0.9, size),
                "like": np.random.uniform(0.05, 0.4, size),
                "dislike": np.random.uniform(0.01, 0.1, size)
            }

        # 3. Apply Multi-Objective Weights to compute base score
        # Weights: Click (0.20), Watch (0.35), Like (0.20), Dislike (-0.10)
        base_scores = (
            0.20 * pred_dict["click"] +
            0.35 * pred_dict["watch_complete"] +
            0.20 * pred_dict["like"] -
            0.10 * pred_dict["dislike"]
        )

        # 4. Fetch dynamic real-time context boosts
        trending_dict = {}
        user_mood = "General"
        if self.stream_processor:
            trending_dict = dict(self.stream_processor.get_trending_items(top_k=20))
            user_mood = self.stream_processor.get_user_current_mood(user_id)

        # 5. Apply context boosts and diversity adjustments
        # To avoid 10 cat videos in a row, we track sequential ranked categories and penalize repeats
        category_repetition_tracker = []
        
        for idx, vid in enumerate(candidate_ids):
            score = float(base_scores[idx])
            
            # Boost A: Social connections (+1.15x boost if watched by friends)
            social_boost = 1.0
            friend_count = 0
            if self.social_graph:
                friend_count = self.social_graph.get_friend_watched_items(user_id).get(vid, 0)
                if friend_count > 0:
                    social_boost = 1.15
                    score *= social_boost
                    
            # Boost B: Trending boost (+1.10x boost if popular)
            trending_boost = 1.0
            if vid in trending_dict:
                trending_boost = 1.10
                score *= trending_boost
                
            # Boost C: Session Mood category match (+1.15x boost)
            v_cat = "General"
            try:
                v_cat = self.videos_df[self.videos_df["video_id"] == vid].iloc[0]["category"]
            except Exception:
                pass
            
            mood_boost = 1.0
            if user_mood == v_cat and user_mood != "General":
                mood_boost = 1.15
                score *= mood_boost

            # Diversity Penalty: check if category dominates last two items
            diversity_penalty = 1.0
            if len(category_repetition_tracker) >= 2:
                if category_repetition_tracker[-1] == v_cat and category_repetition_tracker[-2] == v_cat:
                    diversity_penalty = 0.80 # 20% penalty for category saturation
                    score *= diversity_penalty

            # Update category tracker for diversity
            category_repetition_tracker.append(v_cat)

            # Package scores breakdown for explainability
            breakdown = {
                "p_click": float(pred_dict["click"][idx]),
                "p_watch": float(pred_dict["watch_complete"][idx]),
                "p_like": float(pred_dict["like"][idx]),
                "p_dislike": float(pred_dict["dislike"][idx]),
                "social_boost": social_boost,
                "trending_boost": trending_boost,
                "mood_boost": mood_boost,
                "diversity_penalty": diversity_penalty,
                "friend_watch_count": friend_count
            }
            scored_candidates.append((vid, score, breakdown))

        # Re-sort ranked candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        self.ranking_latencies.append((time.perf_counter() - t0) * 1000)

        return scored_candidates

    def recommend(self, user_id: int, n: int = 20, context: Optional[Dict] = None) -> RecommendationResponse:
        """
        Full two-stage personalized recommendation pipeline:
        1. Retrieve Candidate Pool (500 candidates)
        2. Deep multi-objective Neural Ranking (Top 20 scored list)
        """
        t_start = time.perf_counter()
        self.total_queries += 1

        # Check for cold start user condition
        is_cold = False
        try:
            is_cold = user_id not in self.preprocessor.user_to_idx
        except Exception:
            is_cold = True

        if is_cold and self.cold_start_handler:
            logger.info(f"Cold start detected for user {user_id}. Executing ColdStartHandler fallback.")
            cold_recs = self.cold_start_handler.get_user_fallback(user_id, top_n=n)
            ranked_recs = []
            explanations = {}
            for vid, score in cold_recs:
                v_row = self.videos_df[self.videos_df["video_id"] == vid].iloc[0]
                ranked_recs.append(VideoRecommendation(
                    video_id=vid, score=score,
                    category=v_row["category"], duration=v_row["duration"]
                ))
                explanations[vid] = "Recommended because it is trending globally (Cold Start user)."
                
            elapsed = (time.perf_counter() - t_start) * 1000
            self.total_latencies.append(elapsed)
            
            return RecommendationResponse(
                user_id=user_id,
                group="Control",
                recommendations=ranked_recs,
                cached=False,
                sources_breakdown={"trending": n},
                latency_breakdown={"total": elapsed},
                explanations=explanations,
                is_cold_start=True
            )

        # Stage 1: Ingest candidate pool
        candidates_pool, candidate_sources, sources_breakdown = self.generate_candidates(user_id, context, n=500)

        # Stage 2: Multi-Objective Neural Ranking
        ranked_pool = self.rank_candidates(user_id, candidates_pool, context)

        # Build response models
        final_recs = []
        explanations = {}
        
        # Take Top N
        top_ranked = ranked_pool[:n]
        for vid, score, breakdown in top_ranked:
            v_row = self.videos_df[self.videos_df["video_id"] == vid].iloc[0]
            final_recs.append(VideoRecommendation(
                video_id=vid, score=score,
                category=v_row["category"], duration=v_row["duration"]
            ))
            
            # Construct human-readable explanations based on pipeline rules
            explanations[vid] = self.explain(user_id, vid, breakdown, candidate_sources.get(vid, []))

        # Latency calculations
        elapsed = (time.perf_counter() - t_start) * 1000
        self.total_latencies.append(elapsed)
        
        group = "Treatment"
        if self.ab_engine:
            group = self.ab_engine.get_user_group(user_id)

        # Average latency profile summaries
        latencies = {
            "candidate_generation": sum(np.mean(self.latency_metrics[k]) for k in self.latency_metrics.keys() if "candidate" in k and self.latency_metrics[k]),
            "ranking": np.mean(self.ranking_latencies) if self.ranking_latencies else 0.0,
            "total": elapsed
        }

        return RecommendationResponse(
            user_id=user_id,
            group=group,
            recommendations=final_recs,
            cached=False,
            sources_breakdown=sources_breakdown,
            latency_breakdown=latencies,
            explanations=explanations,
            is_cold_start=False
        )

    def explain(self, user_id: int, item_id: int, breakdown: Optional[Dict] = None, fetched_sources: Optional[List[str]] = None) -> str:
        """
        Returns human-readable explanation based on scoring parameters and retrieval sources.
        """
        if fetched_sources is None:
            fetched_sources = []
            
        bullets = []
        
        # 1. Social watches
        friends_watched = 0
        if breakdown and breakdown.get("friend_watch_count", 0) > 0:
            friends_watched = breakdown["friend_watch_count"]
        elif self.social_graph:
            friends_watched = self.social_graph.get_friend_watched_items(user_id).get(item_id, 0)
            
        if friends_watched > 0:
            bullets.append(f"• {friends_watched} of your friends watched this ✓")

        # 2. Mood category boosts
        user_mood = "General"
        if self.stream_processor:
            user_mood = self.stream_processor.get_user_current_mood(user_id)
            
        v_cat = "General"
        try:
            v_cat = self.videos_df[self.videos_df["video_id"] == item_id].iloc[0]["category"]
        except Exception:
            pass
            
        if user_mood == v_cat and user_mood != "General":
            bullets.append(f"• Matches your current {user_mood} mood ✓")

        # 3. Source indicators
        source_mappings = {
            "cf": "Similar behavioral patterns with users like you watched this",
            "mf": "Latent space similarity matching your video profiles",
            "bert": "Matches sequential patterns in your watch history",
            "trending": "Highly trending in real-time click traffic"
        }
        
        for src in fetched_sources:
            if src in source_mappings:
                bullets.append(f"• {source_mappings[src]} ✓")
                break # Only add one primary retrieval reason for brevity

        if not bullets:
            bullets.append("• High matching probability from MMoE multi-task classification towers ✓")

        return "Recommended because:\n" + "\n".join(bullets)

    def performance_stats(self) -> Dict:
        """
        Returns statistical latency percentiles and cache tracking matrices.
        """
        if not self.total_latencies:
            return {"status": "No queries recorded."}

        p50 = np.percentile(self.total_latencies, 50)
        p95 = np.percentile(self.total_latencies, 95)
        p99 = np.percentile(self.total_latencies, 99)
        
        cache_rate = (self.cache_hits / self.total_queries) if self.total_queries > 0 else 0.0

        gen_stats = {}
        for k in self.latency_metrics.keys():
            if "candidate" in k:
                vals = self.latency_metrics[k]
                gen_stats[k] = float(np.mean(vals)) if vals else 0.0

        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": float(cache_rate),
            "p50_latency_ms": float(p50),
            "p95_latency_ms": float(p95),
            "p99_latency_ms": float(p99),
            "candidate_generation_avg_ms": gen_stats,
            "ranking_avg_ms": float(np.mean(self.ranking_latencies)) if self.ranking_latencies else 0.0
        }
