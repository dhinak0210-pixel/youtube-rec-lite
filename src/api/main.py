import asyncio
import time
from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict
import numpy as np
import pandas as pd

from src.config import TOP_K_CANDIDATES, NUM_USERS, NUM_VIDEOS
from src.utils.logger import logger
from src.data_pipeline.data_loader import load_data
from src.data_pipeline.preprocessors import RecommenderPreprocessor

# Models
from src.models.collaborative_filtering import CollaborativeFilteringRecommender
from src.models.matrix_factorization_als import ALSMatrixFactorization
from src.models.bert4rec import BERT4RecRecommender
from src.models.gnn_recommender import GNNRecommender, UserItemGraph
from src.models.mmoe_ranking import MMoERankingEngine, MultiObjectiveRankingModel
from src.models.recommender_engine import RecommendationEngine

# Modules
from src.cold_start.handler import ColdStartHandler
from src.ab_testing.experiment_engine import ABTestEngine
from src.ab_testing.metrics import ABMetricsEvaluator
from src.streaming.redis_client import MockRedisClient
from src.streaming.simulator import EventQueue, StreamProcessor, StreamEvent

# Schemas
from src.api.schemas import (
    RecommendationRequest, 
    RecommendationResponse, 
    VideoRecommendation, 
    InteractionEvent, 
    DashboardReport, 
    ExperimentMetrics
)

app = FastAPI(
    title="RecoStream API Gateway",
    description="Production-grade two-stage recommendation engine using hybrid deep learning & systems engineering.",
    version="1.0.0"
)

# Global variables holding trained model references and pipeline objects
preprocessor = None
model_cf = None
model_als = None
model_bert = None
model_gnn = None
model_mmoe = None

cold_start_handler = None
ab_engine = None
metrics_evaluator = None
redis_client = None

# Streaming & Orchestration Global Handles
event_queue = None
stream_processor = None
social_graph = None
recommendation_engine = None

users_df = None
videos_df = None
interactions_df = None

# Thread lock to safely update metrics and retrain models asynchronously
state_lock = asyncio.Lock()


@app.on_event("startup")
def startup_event():
    """
    Triggers on FastAPI launch. Sets up data structures, preprocessors, 
    and triggers mock model training.
    """
    global preprocessor, model_cf, model_als, model_bert, model_gnn, model_mmoe
    global cold_start_handler, ab_engine, metrics_evaluator, redis_client
    global users_df, videos_df, interactions_df
    
    logger.info("Starting up RecoStream API Engine...")
    
    # 1. Load Synthetic Data
    import os
    if os.getenv("TESTING") == "True":
        logger.info("TESTING environment detected. Generating mini-sandbox dataset.")
        from src.data_pipeline.data_loader import YouTubeSyntheticDataGenerator
        generator = YouTubeSyntheticDataGenerator(seed=42)
        users_df, videos_df, interactions_df, _ = generator.generate_all(
            num_users=100, num_videos=200, num_interactions=1000, num_follows=100
        )
    else:
        users_df, videos_df, interactions_df = load_data()
    
    # 2. Fit Preprocessing Pipes
    preprocessor = RecommenderPreprocessor()
    preprocessor.fit(users_df, videos_df)
    
    # 3. Train Retrieval Models
    # A. Collaborative Filtering
    model_cf = CollaborativeFilteringRecommender(kind="item", k=15)
    model_cf.fit(interactions_df)
    
    # B. Matrix Factorization (ALS)
    model_als = ALSMatrixFactorization(epochs=10)
    model_als.fit(interactions_df)
    
    # C. BERT4Rec Sequential
    X_bert, y_bert = preprocessor.build_sequential_data(interactions_df)
    vocab_size = len(preprocessor.video_to_idx)
    model_bert = BERT4RecRecommender(vocab_size=vocab_size, epochs=5)
    model_bert.train_model(X_bert, y_bert)
    
    # D. Graph Neural Network (GNN)
    edge_index, v_nodes = preprocessor.build_graph_adjacency(interactions_df)
    num_nodes = len(preprocessor.user_to_idx) + len(preprocessor.video_to_idx)
    model_gnn = GNNRecommender(num_nodes=num_nodes, num_videos=len(preprocessor.video_to_idx), epochs=5)
    model_gnn.train_model(edge_index, v_nodes)
    
    # 4. Train Ranking Model (MMoE)
    X_meta, v_meta = preprocessor.transform_metadata(users_df, videos_df)
    X_rank, y_click, y_watch = preprocessor.build_ranking_features(interactions_df, X_meta, v_meta)
    
    input_dim = X_rank.shape[1] # Feature dim size
    model_mmoe = MMoERankingEngine(input_dim=input_dim, epochs=5)
    model_mmoe.train_model(X_rank, y_click, y_watch)
    
    # 5. Core Modules Setup
    cold_start_handler = ColdStartHandler(users_df, videos_df)
    cold_start_handler.fit(interactions_df)
    
    ab_engine = ABTestEngine()
    metrics_evaluator = ABMetricsEvaluator()
    redis_client = MockRedisClient()
    
    # 6. High-Throughput Real-Time Event Pipeline Setup (Kafka & Flink)
    global event_queue, stream_processor, social_graph, recommendation_engine
    event_queue = EventQueue(maxlen=100000)
    stream_processor = StreamProcessor(event_queue, redis_client)
    
    # Populate item categories into sliding processor
    for _, row in videos_df.iterrows():
        stream_processor.item_categories[int(row["video_id"])] = row["category"]
    
    # Run the stream background daemon thread
    stream_processor.start()
    
    # 7. GNN Social Graph Setup (follows generated from power-law rank distribution)
    social_graph = UserItemGraph(num_users=max(NUM_USERS, len(users_df) + 10), num_items=max(NUM_VIDEOS, len(videos_df) + 10))
    for _, row in interactions_df.iterrows():
        if row["click"] == 1:
            social_graph.add_interaction(int(row["user_id"]), int(row["video_id"]))
    social_graph.generate_synthetic_social_graph(num_connections=500, alpha=1.6)
    
    # 8. Main Recommendation Orchestrator Setup
    engine_config = {
        "cf_model": model_cf,
        "als_model": model_als,
        "bert_model": model_bert,
        "social_graph": social_graph,
        "mmoe_model": model_mmoe,
        "cold_start_handler": cold_start_handler,
        "stream_processor": stream_processor,
        "preprocessor": preprocessor,
        "users_df": users_df,
        "videos_df": videos_df,
        "ab_engine": ab_engine
    }
    recommendation_engine = RecommendationEngine(engine_config)
    
    logger.info("All RecoStream models and Integration Orchestrator trained and cached. System online!")


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Main Personalized Candidate Recommendation Engine endpoint.
    Routes requests according to A/B Testing buckets:
    - Control Group: Collaborative Filtering + Popular fallback.
    - Treatment Group: Hybrid retrieval (CF, ALS, BERT4Rec, GNN) + MMoE Deep Ranking.
    """
    user_id = request.user_id
    top_n = request.top_n
    
    # 1. Check Redis client cache for instant hits
    cached_recs = redis_client.get_cached_recommendations(user_id)
    group = ab_engine.get_user_group(user_id)
    metrics_evaluator.log_impression(group)
    
    if cached_recs:
        # Build cached recommendation response details
        res_list = []
        for vid in cached_recs[:top_n]:
            v_row = videos_df[videos_df["video_id"] == vid].iloc[0]
            res_list.append(VideoRecommendation(
                video_id=vid, score=1.0, 
                category=v_row["category"], duration=v_row["duration"]
            ))
        return RecommendationResponse(user_id=user_id, group=group, recommendations=res_list, cached=True)

    # 2. Check if Cold User
    user_clicks = interactions_df[(interactions_df["user_id"] == user_id) & (interactions_df["click"] == 1)]
    real_time_history = redis_client.get_user_history(user_id)
    
    if len(user_clicks) == 0 and len(real_time_history) == 0:
        # User is cold -> Serve Demographic Popularity
        logger.debug(f"Handling cold start user {user_id}")
        recs = cold_start_handler.get_user_fallback(user_id, top_n)
        res_list = []
        for vid, score in recs:
            v_row = videos_df[videos_df["video_id"] == vid].iloc[0]
            res_list.append(VideoRecommendation(
                video_id=vid, score=score, 
                category=v_row["category"], duration=v_row["duration"]
            ))
        return RecommendationResponse(user_id=user_id, group=group, recommendations=res_list, cached=False)

    # 3. Two-Stage Routing Pipeline via Unified Orchestrator
    response = recommendation_engine.recommend(user_id, n=top_n)
    
    # Cache recommendation IDs in Redis client
    rec_ids = [r.video_id for r in response.recommendations]
    redis_client.set_cached_recommendations(user_id, rec_ids)
    
    return response


@app.post("/interact")
async def interact(event: InteractionEvent):
    """
    Ingests live click stream / watch interaction logs.
    Saves new events to the interactive history queue and updates group experimentation metrics.
    """
    global interactions_df
    
    user_id = event.user_id
    video_id = event.video_id
    click = event.click
    watch_ratio = event.watch_ratio
    like = event.like
    
    # 1. Update real-time sequential cache in Redis and Kafka Event Queue
    if click:
        redis_client.add_to_user_history(user_id, video_id)
        
        # Fetch category context for streaming mood tracking
        v_cat = "General"
        try:
            v_cat = videos_df[videos_df["video_id"] == video_id].iloc[0]["category"]
        except Exception:
            pass
            
        stream_event = StreamEvent(
            user_id=user_id,
            item_id=video_id,
            event_type="click",
            timestamp=time.time(),
            session_id=f"sess_{user_id}",
            watch_percentage=watch_ratio * 100.0,
            context={"category": v_cat}
        )
        if event_queue:
            event_queue.produce(stream_event)
        
    # 2. Record group metrics based on user bucket allocation
    group = ab_engine.get_user_group(user_id)
    metrics_evaluator.log_interaction(group, click, watch_ratio, like)
    
    # 3. Thread-safe log append to in-memory DataFrame
    async with state_lock:

        new_row = {
            "timestamp": pd.Timestamp.now(),
            "user_id": user_id,
            "video_id": video_id,
            "click": click,
            "watch_time_seconds": int(watch_ratio * 120),  # Scale to mock duration
            "watch_percentage": watch_ratio * 100.0,
            "like": like,
            "dislike": 0,
            "share": 0,
            "watch_complete": 1 if watch_ratio >= 0.9 else 0
        }
        interactions_df = pd.concat([interactions_df, pd.DataFrame([new_row])], ignore_index=True)
        
    return {"status": "success", "message": "Interaction event ingested."}

@app.get("/metrics", response_model=DashboardReport)
def get_metrics():
    """
    Fetches online A/B testing reporting statistics for the UI dashboard.
    """
    report = metrics_evaluator.get_report()
    
    metrics_dict = {}
    for group, vals in report.items():
        metrics_dict[group] = ExperimentMetrics(
            group=group,
            impressions=vals["impressions"],
            clicks=vals["clicks"],
            ctr=vals["ctr"],
            avg_watch_ratio=vals["avg_watch_ratio"],
            like_ratio=vals["like_ratio"]
        )
        
    return DashboardReport(
        experiment_name=ab_engine.experiment_name,
        metrics=metrics_dict
    )

@app.post("/reset")
def reset_system():
    """
    Flushes cache, resets evaluation metrics, and re-initializes interaction matrices.
    """
    global interactions_df
    redis_client.clear_cache()
    
    # Re-initialize metrics
    metrics_evaluator.__init__()
    
    # Re-load base synthetic dataset
    _, _, interactions_df = load_data()
    
    return {"status": "success", "message": "Metrics reset and caches cleared."}
