"""
YouTube-scale Recommendation System Command Line Entry Point.

Orchestrates all training phases, inference latency statistics reporting, 
A/B cohort division statistical significance reports, real-time Kafka/Flink
micro-batch event telemetry, and launches the Gradio UI dashboard.
"""

import argparse
import sys
import time
import numpy as np
from collections import defaultdict
from loguru import logger

from config import settings
from training.pipeline import TrainingPipeline
from services.ab_testing import ABTestingService
from streaming.pipeline import EventQueue, StreamProcessor, StreamEvent

# Configure high-fidelity custom Loguru logger formatting
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
)

def run_inference_demo(pipeline: TrainingPipeline):
    """
    Simulates real-time candidate generation and re-ranking for sample users.
    
    Demonstrates candidate retrieval from Item-Based Collaborative Filtering,
    ALS Latent Matrix Factorization, BERT4Rec sequence predictions, and Cold Start popularity.
    """
    print("\n" + "=" * 60)
    print("                 INFERENCE DEMO")
    print("=" * 60)
    
    item_cats = {item.item_id: item.category_id for item in pipeline.items}
    
    # Identify 2 warm users and 1 cold user based on historical click interactions
    warm_users = [u for u in pipeline.users if getattr(u, 'num_interactions', 0) > 20]
    if len(warm_users) < 2:
        sorted_users = sorted(pipeline.users, key=lambda u: getattr(u, 'num_interactions', 0), reverse=True)
        warm_users = sorted_users[:2]
        
    cold_users = [u for u in pipeline.users if getattr(u, 'num_interactions', 0) <= 2]
    if not cold_users:
        sorted_users = sorted(pipeline.users, key=lambda u: getattr(u, 'num_interactions', 0))
        cold_user = sorted_users[0]
    else:
        cold_user = cold_users[0]
        
    demo_users = [warm_users[0], warm_users[1], cold_user]
    latencies = []
    
    for user in demo_users:
        uid = user.user_id
        is_cold = getattr(user, 'num_interactions', 0) <= 2
        
        start_time = time.time()
        candidates = defaultdict(float)
        
        # 1. CF Predictions (score weight 1.0)
        if pipeline.cf:
            try:
                cf_res = pipeline.cf.predict(uid, num_candidates=100)
                for item in cf_res:
                    candidates[item.item_id] += item.score * 1.0
            except Exception as e:
                logger.debug(f"CF prediction failure for user {uid}: {e}")
                
        # 2. MF Predictions (score weight 1.2)
        if pipeline.mf:
            try:
                mf_res = pipeline.mf.predict(uid, num_candidates=100)
                for item in mf_res:
                    candidates[item.item_id] += item.score * 1.2
            except Exception as e:
                logger.debug(f"MF prediction failure for user {uid}: {e}")
                
        # 3. BERT4Rec Predictions (score weight 1.3) if the user has active sequential history
        user_history = []
        if pipeline.cf:
            user_history = list(pipeline.cf.get_history(uid))
        elif pipeline.mf:
            user_history = list(pipeline.mf.user_history.get(uid, []))
            
        if len(user_history) >= 3 and pipeline.bert:
            try:
                bert_res = pipeline.bert.predict(user_history=user_history, top_k=100)
                for item in bert_res:
                    candidates[item.item_id] += item.score * 1.3
            except Exception as e:
                logger.debug(f"BERT prediction failure for user {uid}: {e}")
                
        # 4. Cold Start popularity fallback for cold profiles or empty candidates
        if is_cold or not candidates:
            if pipeline.cold_start:
                try:
                    cold_res = pipeline.cold_start.recommend(user=user, n=100)
                    for item in cold_res:
                        candidates[item.item_id] += item.score * 1.0
                except Exception as e:
                    logger.debug(f"Cold Start prediction failure for user {uid}: {e}")
                    
        # Sort candidates and compile the top 20 recommendations
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:20]
        
        latency = (time.time() - start_time) * 1000.0  # Converted to milliseconds
        latencies.append(latency)
        
        user_type = "COLD" if is_cold else "WARM"
        logger.info(f"User ID: {uid:<4} | Type: {user_type:<4} | Interactions: {getattr(user, 'num_interactions', 0):<3} | Latency: {latency:.2f}ms")
        
        # Display top 5 recommendations with metadata
        top_5 = sorted_candidates[:5]
        for rank, (item_id, score) in enumerate(top_5, 1):
            cat_id = item_cats.get(item_id, "Unknown")
            logger.info(f"   Rank {rank}: Video ID {item_id:<5} | Category: {cat_id:<3} | Score: {score:.4f}")
            
    mean_lat = np.mean(latencies)
    p99_lat = np.percentile(latencies, 99)
    logger.info(f"Latency Statistics: Mean = {mean_lat:.2f}ms, P99 = {p99_lat:.2f}ms")

def run_ab_demo():
    """
    Performs deterministic user partition cohort assignments and generates reports.
    
    Performs two-proportion Z-tests to identify lifts, standard errors, and verdicts.
    """
    print("\n" + "=" * 60)
    print("               A/B TEST SIMULATION")
    print("=" * 60)
    
    ab_service = ABTestingService()
    report = ab_service.simulate(
        exp_id="ranking_v2_vs_v1",
        n_users=3000,
        ctrl_ctr=0.050,
        treat_ctr=0.062,
        ctrl_completion=0.150,
        treat_completion=0.180
    )
    
    ctrl = report["control"]
    treat = report["treatment"]
    
    logger.info(f"Control group   | Impressions: {ctrl['impressions']:<4} | CTR: {ctrl['ctr']:.4%}, Completion Rate: {ctrl['completion_rate']:.4%}")
    logger.info(f"Treatment group | Impressions: {treat['impressions']:<4} | CTR: {treat['ctr']:.4%}, Completion Rate: {treat['completion_rate']:.4%}")
    
    if "ctr_test" in report:
        ctr_t = report["ctr_test"]
        logger.info(f"CTR Z-Test     | Z-Stat: {ctr_t['z_stat']:.4f} | P-Value: {ctr_t['p_value']:.4e} | Lift: {ctr_t['relative_lift']:.2%} | Verdict: {ctr_t['verdict']}")
        
    if "completion_test" in report:
        comp_t = report["completion_test"]
        logger.info(f"Completion Z-Test | Lift: {comp_t['relative_lift']:.2%} | Verdict: {comp_t['verdict']}")

def run_streaming_demo(pipeline: TrainingPipeline):
    """
    Simulates high-velocity Apache Kafka stream ingest and background Flink window processing.
    """
    print("\n" + "=" * 60)
    print("            REAL-TIME STREAMING DEMO")
    print("=" * 60)
    
    queue = EventQueue()
    processor = StreamProcessor(queue=queue, feature_store=pipeline.feature_store)
    
    item_cats = {item.item_id: item.category_id for item in pipeline.items}
    user_ids = [u.user_id for u in pipeline.users]
    item_ids = [i.item_id for i in pipeline.items]
    
    processor.start()
    
    start_time = time.time()
    
    # Ingest 2000 random user interaction clicks in parallel
    for _ in range(2000):
        iid = int(np.random.choice(item_ids))
        event = StreamEvent(
            user_id=int(np.random.choice(user_ids)),
            item_id=iid,
            event_type=str(np.random.choice(["impression", "click", "watch_complete", "like"])),
            timestamp=time.time(),
            watch_percentage=float(np.random.uniform(0.0, 1.0)),
            category_id=int(item_cats.get(iid, 0))
        )
        queue.produce(event)
        
    # Wait for background thread executor processing completion
    time.sleep(0.5)
    
    elapsed_time = time.time() - start_time
    st = processor.stats()
    processed = st["processed"]
    
    throughput = processed / elapsed_time
    
    logger.info(f"Events Produced : 2000")
    logger.info(f"Events Processed: {processed}")
    logger.info(f"Throughput      : {throughput:.2f} events/sec")
    logger.info(f"Active Users    : {st['active_users']}")
    
    # Output trending items inside active sliding windows
    trending = processor.get_trending(top_k=5)
    logger.info("Top 5 Trending Videos:")
    for rank, (item_id, count) in enumerate(trending, 1):
        cat_id = item_cats.get(item_id, "Unknown")
        logger.info(f"   Rank {rank}: Video ID {item_id:<5} | Category: {cat_id:<3} | Views: {count}")
        
    processor.stop()

def main():
    """Parses arguments, executes orchestrator pipeline, and launches dashboard UI."""
    parser = argparse.ArgumentParser(description="RecoStream AI Command Line Entry Point.")
    parser.add_argument("--quick", action="store_true", help="Launch pipeline using lightweight mock dataset scale.")
    parser.add_argument("--demo", action="store_true", help="Launches interactive Gradio web application dashboard.")
    args = parser.parse_args()
    
    # Initialize and execute training pipeline stages
    pipeline = TrainingPipeline()
    results = pipeline.run(quick=args.quick)
    
    logger.info(f"Offline Training Pipeline completed in {results['elapsed_seconds']:.2f} seconds.")
    
    # Run command-line simulations
    run_inference_demo(pipeline)
    run_ab_demo()
    run_streaming_demo(pipeline)
    
    # Launch UI Dashboard
    if args.demo:
        logger.info("Launching Gradio demo dashboard...")
        from demo.app import launch_demo
        launch_demo(pipeline)

if __name__ == "__main__":
    main()
