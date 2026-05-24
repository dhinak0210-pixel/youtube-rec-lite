# ---
# title: YouTube Recommendation Lite
# emoji: 🎬
# colorFrom: red
# colorTo: purple  
# sdk: gradio
# sdk_version: 4.7.0
# python_version: 3.10
# app_file: hf_app.py
# pinned: false
# ---

"""
Hugging Face Spaces Standalone Web Application.
"""

import sys
sys.path.insert(0, '.')

import os
import time
import numpy as np
import gradio as gr
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

from config import settings
from data import UserProfile, ItemProfile
from training.pipeline import TrainingPipeline
from services.ab_testing import ABTestingService
from streaming.pipeline import EventQueue, StreamProcessor, StreamEvent
from evaluation.metrics import Metrics

# Global pipeline instance reference
_pipeline: Optional[TrainingPipeline] = None

# Pre-train globally on script import
try:
    print("⚙️ Training models, please wait...")
    _pipeline = TrainingPipeline(num_users=200, num_items=800, num_interactions=8000)
    _pipeline.run(quick=True)
    print("✅ Training complete! Ready to serve...")
except Exception as e:
    print(f"❌ Error during automatic pre-training: {e}")
    _pipeline = None

def _get_pipeline() -> TrainingPipeline:
    """Retrieves the pre-trained pipeline or raises an informative exception if training failed."""
    global _pipeline
    if _pipeline is None:
        raise ValueError("Model pipeline is not trained or initialized. Please check logs for errors.")
    return _pipeline

def get_recommendations(user_id: int, n_recs: int, hour: int, device: str) -> Tuple[str, List[List[Any]]]:
    """Retrieves candidates and scores them for personalized user recommendations."""
    try:
        pipeline = _get_pipeline()
        item_cats = {item.item_id: item.category_id for item in pipeline.items}
        num_users = len(pipeline.users)
        
        # Map user id within modulo bounds of active generated users
        uid = int(user_id) % max(num_users, 1)
        user = pipeline.users[uid]
        
        start_time = time.time()
        candidates = defaultdict(lambda: {"score": 0.0, "source": "None"})
        
        # 1. CF Candidates (score weight 1.0)
        if pipeline.cf:
            try:
                cf_res = pipeline.cf.predict(uid, num_candidates=50)
                for item in cf_res:
                    candidates[item.item_id]["score"] += item.score * 1.0
                    candidates[item.item_id]["source"] = "CF"
            except Exception:
                pass
                
        # 2. MF Candidates (score weight 1.2)
        if pipeline.mf:
            try:
                mf_res = pipeline.mf.predict(uid, num_candidates=50)
                for item in mf_res:
                    candidates[item.item_id]["score"] += item.score * 1.2
                    if candidates[item.item_id]["source"] in ["None", "CF"]:
                        candidates[item.item_id]["source"] = "MF"
            except Exception:
                pass
                
        # 3. BERT4Rec Candidates (score weight 1.3) if the user has sequential history
        user_history = []
        if pipeline.cf:
            user_history = list(pipeline.cf.get_history(uid))
        elif pipeline.mf:
            user_history = list(pipeline.mf.user_history.get(uid, []))
            
        if len(user_history) >= 3 and pipeline.bert:
            try:
                bert_res = pipeline.bert.predict(user_history=user_history, top_k=50)
                for item in bert_res:
                    candidates[item.item_id]["score"] += item.score * 1.3
                    candidates[item.item_id]["source"] = "BERT4Rec"
            except Exception:
                pass
                
        # 4. Cold Start heuristic fallback for new profiles or empty lists
        is_cold = getattr(user, 'num_interactions', 0) <= 5
        if is_cold or not candidates:
            if pipeline.cold_start:
                try:
                    cold_res = pipeline.cold_start.recommend(user=user, n=50)
                    for item in cold_res:
                        candidates[item.item_id]["score"] += item.score * 1.0
                        candidates[item.item_id]["source"] = "ColdStart"
                except Exception:
                    pass
                    
        # Sort and take top recommendations
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)[:int(n_recs)]
        
        latency = (time.time() - start_time) * 1000.0  # in ms
        summary = f"**User ID**: `{uid}` | **Cold Start**: `{'Yes 🆕' if is_cold else 'No 🔥'}` | **Interactions Count**: `{getattr(user, 'num_interactions', 0)}` | **Latency**: `{latency:.2f}ms`"
        
        rows = []
        for rank, (item_id, item_data) in enumerate(sorted_candidates, 1):
            cat_id = item_cats.get(item_id, 0)
            rows.append([
                rank,
                item_id,
                f"Category {cat_id}",
                round(item_data["score"], 4),
                item_data["source"]
            ])
            
        return summary, rows
    except Exception as e:
        return f"⚠️ Error processing recommendations: {str(e)}", []

def run_ab_test(n_users: int, ctrl_ctr: float, treat_ctr: float) -> Tuple[List[List[Any]], str]:
    """Runs A/B Testing cohort simulation and generates two-proportion Z-test report."""
    try:
        ab_service = ABTestingService()
        report = ab_service.simulate(
            exp_id="dashboard_ab_test",
            n_users=int(n_users),
            ctrl_ctr=float(ctrl_ctr),
            treat_ctr=float(treat_ctr),
            ctrl_completion=0.150,
            treat_completion=0.180
        )
        
        ctrl = report["control"]
        treat = report["treatment"]
        
        summary_rows = [
            ["Control Group", ctrl["impressions"], f"{ctrl['ctr']:.4%}", f"{ctrl['completion_rate']:.4%}"],
            ["Treatment Group", treat["impressions"], f"{treat['ctr']:.4%}", f"{treat['completion_rate']:.4%}"]
        ]
        
        ctr_t = report.get("ctr_test", {"z_stat": 0.0, "p_value": 1.0, "significant": False, "relative_lift": 0.0, "verdict": "CONTINUE"})
        comp_t = report.get("completion_test", {"relative_lift": 0.0, "verdict": "CONTINUE"})
        
        stat_text = f"""
### 🔬 Statistical Hypothesis Z-Test Significance Report

* **Click-Through Rate (CTR) Z-Test**:
  * **Z-Statistic**: `{ctr_t['z_stat']:.4f}`
  * **P-Value**: `{ctr_t['p_value']:.4e}`
  * **Statistically Significant**: `{'Yes ✅' if ctr_t['significant'] else 'No ❌'}`
  * **Relative Conversion Lift**: `{ctr_t['relative_lift']:.2%}`
  * **Decision Action Verdict**: **`{ctr_t['verdict']}`**

* **Watch Completion Rate Z-Test**:
  * **Relative Completion Lift**: `{comp_t['relative_lift']:.2%}`
  * **Decision Action Verdict**: **`{comp_t['verdict']}`**
"""
        return summary_rows, stat_text
    except Exception as e:
        return [], f"⚠️ Error running A/B Test simulation: {str(e)}"

def compare_models() -> List[List[Any]]:
    """Calculates NDCG@10, Precision@10, Hit Rate, and Catalog Diversity metrics across models."""
    try:
        pipeline = _get_pipeline()
        
        # Build ground truth
        test_truth = defaultdict(set)
        for inter in pipeline.test_interactions:
            if inter.weight > 1.0:
                test_truth[inter.user_id].add(inter.item_id)
                
        item_cats = {item.item_id: item.category_id for item in pipeline.items}
        eval_users = [uid for uid, item_set in test_truth.items() if len(item_set) >= 2][:100]
        
        cf_metrics_list = []
        mf_metrics_list = []
        
        for uid in eval_users:
            rel = test_truth[uid]
            
            # Evaluate CF
            cf_recs = []
            if pipeline.cf:
                try:
                    cf_recs = [item.item_id for item in pipeline.cf.predict(uid, num_candidates=20)]
                except Exception:
                    pass
            cf_metrics = Metrics.all_metrics(cf_recs, rel, k=10, item_cats=item_cats)
            cf_metrics_list.append(cf_metrics)
            
            # Evaluate MF
            mf_recs = []
            if pipeline.mf:
                try:
                    mf_recs = [item.item_id for item in pipeline.mf.predict(uid, num_candidates=20)]
                except Exception:
                    pass
            mf_metrics = Metrics.all_metrics(mf_recs, rel, k=10, item_cats=item_cats)
            mf_metrics_list.append(mf_metrics)
            
        rows = []
        if cf_metrics_list:
            rows.append([
                "Collaborative Filtering",
                round(np.mean([m.get("ndcg@10", 0.0) for m in cf_metrics_list]), 4),
                round(np.mean([m.get("precision@10", 0.0) for m in cf_metrics_list]), 4),
                round(np.mean([m.get("hitrate@10", 0.0) for m in cf_metrics_list]), 4),
                round(np.mean([m.get("diversity@10", 0.0) for m in cf_metrics_list]), 4)
            ])
        else:
            rows.append(["Collaborative Filtering", 0.0, 0.0, 0.0, 0.0])
            
        if mf_metrics_list:
            rows.append([
                "Matrix Factorization (ALS)",
                round(np.mean([m.get("ndcg@10", 0.0) for m in mf_metrics_list]), 4),
                round(np.mean([m.get("precision@10", 0.0) for m in mf_metrics_list]), 4),
                round(np.mean([m.get("hitrate@10", 0.0) for m in mf_metrics_list]), 4),
                round(np.mean([m.get("diversity@10", 0.0) for m in mf_metrics_list]), 4)
            ])
        else:
            rows.append(["Matrix Factorization (ALS)", 0.0, 0.0, 0.0, 0.0])
            
        return rows
    except Exception as e:
        print(f"⚠️ Error comparing models: {e}")
        return [["Error Comparing Models", 0.0, 0.0, 0.0, 0.0]]

def run_streaming_sim(n_events: int) -> Tuple[str, List[List[Any]]]:
    """Ingests clickstream events in a real-time event queue and evaluates micro-batch lag."""
    try:
        pipeline = _get_pipeline()
        
        queue = EventQueue()
        processor = StreamProcessor(queue=queue, feature_store=pipeline.feature_store)
        
        item_cats = {item.item_id: item.category_id for item in pipeline.items}
        user_ids = [u.user_id for u in pipeline.users]
        item_ids = [i.item_id for i in pipeline.items]
        
        processor.start()
        
        start_time = time.time()
        
        for _ in range(int(n_events)):
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
            
        # Wait for micro-batch processing execution
        time.sleep(0.3)
        
        elapsed_time = time.time() - start_time
        st = processor.stats()
        processed = st["processed"]
        
        throughput = processed / elapsed_time
        
        summary_md = f"""
### ⚡ Real-Time Ingestion Throughput Statistics

* **Events Ingested**: `{n_events}`
* **Events Processed**: `{processed}`
* **Flink Ingestion Throughput**: `{throughput:.2f} events/sec`
* **Active User Sessions**: `{st['active_users']}`
"""
        trending = processor.get_trending(top_k=10)
        trend_rows = []
        for rank, (item_id, count) in enumerate(trending, 1):
            cat_id = item_cats.get(item_id, 0)
            trend_rows.append([rank, item_id, f"Category {cat_id}", count])
            
        processor.stop()
        return summary_md, trend_rows
    except Exception as e:
        return f"⚠️ Error in Streaming Ingestion: {str(e)}", []

def cold_vs_warm() -> Tuple[str, List[List[Any]], str, List[List[Any]]]:
    """Retrieves and displays recommendation comparisons between cold and warm user profiles."""
    try:
        pipeline = _get_pipeline()
        item_cats = {item.item_id: item.category_id for item in pipeline.items}
        
        # 1. Cold user profile
        cold_users = [u for u in pipeline.users if getattr(u, 'num_interactions', 0) < 5]
        if not cold_users:
            sorted_users = sorted(pipeline.users, key=lambda u: getattr(u, 'num_interactions', 0))
            cold_user = sorted_users[0]
        else:
            cold_user = cold_users[0]
            
        # 2. Warm user profile
        warm_users = [u for u in pipeline.users if getattr(u, 'num_interactions', 0) > 50]
        if not warm_users:
            sorted_users = sorted(pipeline.users, key=lambda u: getattr(u, 'num_interactions', 0), reverse=True)
            warm_user = sorted_users[0]
        else:
            warm_user = warm_users[0]
            
        def get_recs_for_user(user) -> List[List[Any]]:
            uid = user.user_id
            is_cold = getattr(user, 'num_interactions', 0) < 5
            candidates = defaultdict(float)
            
            # CF
            if pipeline.cf:
                try:
                    cf_res = pipeline.cf.predict(uid, num_candidates=50)
                    for item in cf_res:
                        candidates[item.item_id] += item.score * 1.0
                except Exception:
                    pass
                    
            # MF
            if pipeline.mf:
                try:
                    mf_res = pipeline.mf.predict(uid, num_candidates=50)
                    for item in mf_res:
                        candidates[item.item_id] += item.score * 1.2
                except Exception:
                    pass
                    
            # BERT4Rec
            user_history = []
            if pipeline.cf:
                user_history = list(pipeline.cf.get_history(uid))
            elif pipeline.mf:
                user_history = list(pipeline.mf.user_history.get(uid, []))
                
            if len(user_history) >= 3 and pipeline.bert:
                try:
                    bert_res = pipeline.bert.predict(user_history=user_history, top_k=50)
                    for item in bert_res:
                        candidates[item.item_id] += item.score * 1.3
                except Exception:
                    pass
                    
            # Cold start fallback
            if is_cold or not candidates:
                if pipeline.cold_start:
                    try:
                        cold_res = pipeline.cold_start.recommend(user=user, n=50)
                        for item in cold_res:
                            candidates[item.item_id] += item.score * 1.0
                    except Exception:
                        pass
                        
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:10]
            rows = []
            for rank, (item_id, score) in enumerate(sorted_cands, 1):
                cat_id = item_cats.get(item_id, 0)
                rows.append([rank, item_id, f"Category {cat_id}", round(score, 4)])
            return rows
            
        cold_rows = get_recs_for_user(cold_user)
        warm_rows = get_recs_for_user(warm_user)
        
        cold_info_md = f"""
### 🆕 Cold Start Profile
* **User ID**: `{cold_user.user_id}`
* **Historical Engagements**: `{getattr(cold_user, 'num_interactions', 0)}`
* **Profile Cohort**: `Cold Profile`
"""
        warm_info_md = f"""
### 🔥 Warm Profile
* **User ID**: `{warm_user.user_id}`
* **Historical Engagements**: `{getattr(warm_user, 'num_interactions', 0)}`
* **Profile Cohort**: `Active User`
"""
        return cold_info_md, cold_rows, warm_info_md, warm_rows
    except Exception as e:
        return f"⚠️ Error comparing profile classes: {str(e)}", [], "", []

# Custom dark-theme CSS layout properties
custom_theme_css = """
body { background-color: #0f111a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
.gradio-container { background: radial-gradient(circle at top, #1a1d30, #0f111a); border-radius: 12px; }
.tabs { border-bottom: 2px solid #2d3748 !important; }
.tab-button-active { color: #63b3ed !important; border-bottom-color: #63b3ed !important; }
button.primary { background: linear-gradient(135deg, #3182ce, #63b3ed) !important; color: white !important; font-weight: bold !important; border: none !important; }
button.primary:hover { background: linear-gradient(135deg, #2b6cb0, #4299e1) !important; }
"""

def build_app() -> gr.Blocks:
    """Compiles the Gradio web elements and layouts."""
    with gr.Blocks(title="YouTube Recommendation Lite 🎬", theme=gr.themes.Soft(), css=custom_theme_css) as app:
        gr.Markdown("""
# 🎬 YouTube Recommendation Lite Dashboard
This interactive dashboard demonstrates candidate retrieval, deep ranking, A/B telemetry, and micro-batch pipelines in real time under zero cost!
""")
        
        # Tab 1: personalized recommendation scorer
        with gr.Tab("🎯 Get Recommendations"):
            gr.Markdown("### Query Candidates and Multi-Model Re-Ranking Scores")
            with gr.Row():
                with gr.Column():
                    user_id_input = gr.Slider(0, 199, value=42, step=1, label="User ID")
                    n_recs_input = gr.Slider(5, 20, value=10, step=1, label="Recommendations Count")
                    hour_input = gr.Slider(0, 23, value=12, step=1, label="Context Hour")
                    device_input = gr.Dropdown(["Mobile Phone", "Desktop Computer", "Smart TV"], value="Mobile Phone", label="Context Device")
                    get_rec_btn = gr.Button("🚀 Get Recommendations!", variant="primary")
                with gr.Column():
                    rec_summary_output = gr.Markdown("Please hit the button to trigger recommendation scores...")
                    rec_table_output = gr.Dataframe(
                        headers=["Rank", "Video ID", "Category", "Relevance Score", "Model Source"],
                        datatype=["number", "number", "str", "number", "str"],
                        interactive=False
                    )
            get_rec_btn.click(
                fn=get_recommendations,
                inputs=[user_id_input, n_recs_input, hour_input, device_input],
                outputs=[rec_summary_output, rec_table_output]
            )
            
        # Tab 2: A/B Significance test simulation
        with gr.Tab("🔬 A/B Testing"):
            gr.Markdown("### Simulate conversion distributions and calculate statistical two-proportion Z-tests")
            with gr.Row():
                with gr.Column():
                    ab_users_input = gr.Slider(500, 5000, value=3000, step=100, label="Cohort User Count")
                    ctrl_ctr_input = gr.Slider(0.01, 0.15, value=0.05, step=0.005, label="Control Group CTR")
                    treat_ctr_input = gr.Slider(0.01, 0.15, value=0.062, step=0.005, label="Treatment Group CTR")
                    run_ab_btn = gr.Button("▶️ Run A/B Test", variant="primary")
                with gr.Column():
                    ab_summary_output = gr.Dataframe(
                        headers=["Experiment Group", "Cohort Count", "CTR Conversion", "Completion Rate"],
                        datatype=["str", "number", "str", "str"],
                        interactive=False
                    )
                    ab_stats_output = gr.Markdown("Please run A/B test simulation to see analytical p-value statistics reports...")
            run_ab_btn.click(
                fn=run_ab_test,
                inputs=[ab_users_input, ctrl_ctr_input, treat_ctr_input],
                outputs=[ab_summary_output, ab_stats_output]
            )
            
        # Tab 3: Offline Model Comparisons
        with gr.Tab("📊 Model Comparison"):
            gr.Markdown("### Compute offline accuracy evaluations across Test dataset splits")
            compare_btn = gr.Button("📈 Compare All Models", variant="primary")
            compare_output = gr.Dataframe(
                headers=["Retrieval Model", "NDCG@10", "Precision@10", "Hit Rate@10", "Category Diversity"],
                datatype=["str", "number", "number", "number", "number"],
                interactive=False
            )
            compare_btn.click(
                fn=compare_models,
                inputs=[],
                outputs=[compare_output]
            )
            
        # Tab 4: Apache Kafka/Flink Ingestion Pipeline
        with gr.Tab("⚡ Real-Time Streaming"):
            gr.Markdown("### Ingest click-stream event sequences and monitor real-time aggregates")
            with gr.Row():
                with gr.Column():
                    n_events_input = gr.Slider(100, 5000, value=1000, step=100, label="Events Batch Size")
                    run_stream_btn = gr.Button("▶️ Run Streaming Simulation", variant="primary")
                with gr.Column():
                    stream_summary_output = gr.Markdown("Please trigger events simulation...")
                    stream_trend_output = gr.Dataframe(
                        headers=["Rank", "Video ID", "Category", "Active Views Count"],
                        datatype=["number", "number", "str", "number"],
                        interactive=False
                    )
            run_stream_btn.click(
                fn=run_streaming_sim,
                inputs=[n_events_input],
                outputs=[stream_summary_output, stream_trend_output]
            )
            
        # Tab 5: Cold Start vs Warm User comparisons
        with gr.Tab("🆕 Cold Start vs Warm User"):
            gr.Markdown("### Benchmark recommendation accuracy differences between Cold Start profiles vs Active users")
            compare_cw_btn = gr.Button("🔍 Compare Cold vs Warm", variant="primary")
            with gr.Row():
                with gr.Column():
                    cold_summary_output = gr.Markdown("### 🆕 Cold Profile")
                    cold_table_output = gr.Dataframe(
                        headers=["Rank", "Video ID", "Category", "Score"],
                        datatype=["number", "number", "str", "number"],
                        interactive=False
                    )
                with gr.Column():
                    warm_summary_output = gr.Markdown("### 🔥 Warm Profile")
                    warm_table_output = gr.Dataframe(
                        headers=["Rank", "Video ID", "Category", "Score"],
                        datatype=["number", "number", "str", "number"],
                        interactive=False
                    )
            compare_cw_btn.click(
                fn=cold_vs_warm,
                inputs=[],
                outputs=[cold_summary_output, cold_table_output, warm_summary_output, warm_table_output]
            )
            
        gr.Markdown("---")
        gr.Markdown("<center>Built with PyTorch • Gradio | Zero Cost 🆓</center>")
        
    return app

# Initialize and launch Gradio blocks app standalone without hardcoded arguments
app = build_app()

if __name__ == "__main__":
    app.launch()
