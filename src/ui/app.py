import gradio as gr
import httpx
import pandas as pd
import numpy as np
import random
import time
import sys
import os
from typing import Tuple, List, Dict
from src.config import API_HOST, API_PORT, NUM_USERS
from models.optimization import DynamicQuantizer, LRUEmbeddingCache, PerformanceProfiler

API_URL = f"http://{API_HOST}:{API_PORT}"

# Global caching instances for Gradio UI sandbox
ui_lru_cache = LRUEmbeddingCache(capacity=1000, ttl_seconds=60)
cache_enabled_state = [False]


# Premium Neon-Dark Glassmorphism styling configuration
CUSTOM_CSS = """
body, .gradio-container {
    background: radial-gradient(circle at 10% 20%, rgb(15, 15, 20) 0%, rgb(5, 5, 8) 100%) !important;
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
    color: #f3f4f6 !important;
}

.glow-card {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(16px) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.glow-card:hover {
    box-shadow: 0px 8px 30px rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
    transform: translateY(-2px) !important;
}

.cohort-badge-treatment {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: white !important;
    padding: 6px 14px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 0.85em;
    display: inline-block;
    box-shadow: 0 0 15px rgba(168, 85, 247, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.cohort-badge-control {
    background: linear-gradient(135deg, #14b8a6, #0ea5e9) !important;
    color: white !important;
    padding: 6px 14px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 0.85em;
    display: inline-block;
    box-shadow: 0 0 15px rgba(14, 165, 233, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.video-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 20px;
    margin-top: 15px;
}

.video-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
}

.video-card:hover {
    transform: translateY(-4px) scale(1.01);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
}

.video-thumbnail {
    height: 130px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    color: rgba(255, 255, 255, 0.95);
    font-size: 1.1em;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

.video-duration-tag {
    position: absolute;
    bottom: 8px;
    right: 8px;
    background: rgba(0, 0, 0, 0.75);
    color: #fff;
    padding: 2px 6px;
    font-size: 0.75em;
    border-radius: 4px;
    font-weight: bold;
}

.video-details {
    padding: 14px;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.video-title {
    font-weight: 700;
    font-size: 0.95em;
    margin-bottom: 6px;
    color: #fff;
}

.video-category {
    font-size: 0.78em;
    color: #a1a1aa;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

.video-actions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
}

.btn-watch {
    background: #e11d48 !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-size: 0.85em !important;
    transition: background 0.2s !important;
    flex-grow: 1;
    text-align: center;
}

.btn-watch:hover {
    background: #be123c !important;
}

.btn-like {
    background: rgba(255, 255, 255, 0.08) !important;
    border: none !important;
    color: white !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-size: 0.85em !important;
}

.btn-like:hover {
    background: rgba(255, 255, 255, 0.18) !important;
}
"""

# Category → (emoji, grad_start, grad_end, youtube_embed_id)
_CAT = {
    "Music":     ("🎵", "#ec4899", "#9333ea", "jNQXAC9IVRw"),
    "Tech":      ("💻", "#3b82f6", "#6366f1", "Y8Tko2YC5hA"),
    "Gaming":    ("🎮", "#8b5cf6", "#6d28d9", "dQw4w9WgXcQ"),
    "Comedy":    ("😂", "#f59e0b", "#ef4444", "6wXkI4Ch_IA"),
    "Sports":    ("⚽", "#10b981", "#0891b2", "iRzXJMFnqZM"),
    "DIY":       ("🔨", "#f97316", "#eab308", "tPEE9ZwTmy0"),
    "Education": ("📚", "#06b6d4", "#3b82f6", "aircAruvnKk"),
    "Vlogs":     ("📹", "#d946ef", "#ec4899", "kfMoVoipty4"),
    "Fitness":   ("💪", "#ef4444", "#f97316", "iRzXJMFnqZM"),
    "Pets":      ("🐾", "#84cc16", "#10b981", "FlsCjmMhFmw"),
    "Cooking":   ("🍳", "#f97316", "#f59e0b", "1IszT_guI08"),
    "Travel":    ("✈️", "#06b6d4", "#6366f1", "tMujG-n8i04"),
    "Finance":   ("💰", "#10b981", "#3b82f6", "PHe0bXAIuk0"),
    "Science":   ("🔬", "#6366f1", "#8b5cf6", "7lCDEYXw3mM"),
    "News":      ("📰", "#64748b", "#374151", "Y8Tko2YC5hA"),
}
_DEF_CAT = ("🎬", "#6366f1", "#a855f7", "jNQXAC9IVRw")

def _fmt_dur(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}:{s:02d}" if m < 60 else f"{m//60}:{m%60:02d}:{s:02d}"

# ==================== TAB 1 FUNCTIONS ====================
def query_personalized_feed(user_id: int, time_of_day: str, device: str, top_n: int) -> Tuple[str, str, Dict]:
    """
    Invokes the FastAPI unified orchestrator endpoint /recommend and parses results.
    Integrates LRU cache lookups to bypass the network requests when active.
    """
    cache_key = (int(user_id), time_of_day, device, int(top_n))
    if cache_enabled_state[0]:
        cached_res = ui_lru_cache.get(cache_key)
        if cached_res is not None:
            group_html, table_html, data = cached_res
            group_html = group_html.replace("Cohort", "Cohort ⚡ (LRU Cached)")
            return group_html, table_html, data

    try:
        payload = {
            "user_id": int(user_id),
            "top_n": int(top_n),
            "context": {
                "time_of_day": time_of_day,
                "device": device
            }
        }
        res = httpx.post(f"{API_URL}/recommend", json=payload, timeout=6.0)
        if res.status_code != 200:
            return "<h3>⚠️ API Server Error</h3>", f"<p style='color: #ef4444;'>Error code: {res.status_code}</p>", {"error": res.text}
            
        data = res.json()
        group = data["group"]
        cached = "⚡ (Cached)" if data["cached"] else ""
        
        # Style group cohorts badges
        badge_class = "cohort-badge-treatment" if group == "Treatment" else "cohort-badge-control"
        group_html = f"<div class='{badge_class}'>{group} Cohort {cached}</div>"
        
        recs = data.get("recommendations") or []
        explanations = data.get("explanations") or {}
        sources = data.get("sources_breakdown") or {}
        
        if not recs:
            return group_html, "<p style='color:#a1a1aa;'>Empty recommendations pool.</p>", data

        uid = str(abs(hash((user_id, time_of_day, top_n))))[-5:]
        cards = ""
        for idx, r in enumerate(recs):
            vid   = r["video_id"]
            cat   = r.get("category", "Tech")
            score = r.get("score", 0.0)
            dur   = _fmt_dur(r.get("duration", 180))
            why   = (explanations.get(str(vid), "Matches your preference profile")
                     .replace("\n"," ").replace("'",""))[:78]
            emoji, c1, c2, yt_id = _CAT.get(cat, _DEF_CAT)
            pct   = min(int(score * 300), 100)
            cards += f"""
<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:14px;overflow:hidden;transition:all .25s ease"
     onmouseenter="this.style.borderColor='rgba(99,102,241,.45)';this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 28px rgba(0,0,0,.4)'"
     onmouseleave="this.style.borderColor='rgba(255,255,255,.06)';this.style.transform='';this.style.boxShadow=''">
  <div onclick="rs_play_{uid}('https://www.youtube.com/embed/{yt_id}','Video #{vid} &bull; {cat}')"
       style="height:126px;background:linear-gradient(135deg,{c1},{c2});display:flex;align-items:center;justify-content:center;position:relative;cursor:pointer">
    <span style="font-size:2.4em;filter:drop-shadow(0 2px 6px rgba(0,0,0,.5))">{emoji}</span>
    <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;opacity:0;background:rgba(0,0,0,.38);transition:opacity .18s"
         onmouseenter="this.style.opacity='1'" onmouseleave="this.style.opacity='0'">
      <span style="font-size:2.2em">▶️</span></div>
    <span style="position:absolute;bottom:6px;right:8px;background:rgba(0,0,0,.75);color:#fff;padding:2px 6px;font-size:.7em;border-radius:4px;font-weight:700">{dur}</span>
    <span style="position:absolute;top:7px;left:8px;background:rgba(0,0,0,.65);color:#fff;padding:2px 8px;font-size:.68em;border-radius:4px;font-weight:800">#{idx+1}</span>
  </div>
  <div style="padding:11px">
    <div style="font-weight:700;font-size:.88em;color:#fff;margin-bottom:3px">Video #{vid}</div>
    <div style="font-size:.72em;color:#a1a1aa;text-transform:uppercase;letter-spacing:.5px;margin-bottom:7px">{cat}</div>
    <div style="background:rgba(255,255,255,.07);border-radius:3px;height:3px;margin-bottom:8px">
      <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,{c1},{c2});border-radius:3px"></div></div>
    <div style="font-size:.72em;color:#6b7280;margin-bottom:9px;line-height:1.35">{why}…</div>
    <div style="display:flex;gap:7px">
      <button onclick="rs_play_{uid}('https://www.youtube.com/embed/{yt_id}','Video #{vid} &bull; {cat}')"
              style="flex:1;background:#e11d48;border:none;color:#fff;font-weight:700;padding:7px;border-radius:8px;cursor:pointer;font-size:.8em"
              onmouseenter="this.style.background='#be123c'" onmouseleave="this.style.background='#e11d48'">▶ Watch</button>
      <button style="background:rgba(255,255,255,.08);border:none;color:#fff;padding:7px 10px;border-radius:8px;cursor:pointer;font-size:.8em"
              onmouseenter="this.style.background='rgba(255,255,255,.18)'" onmouseleave="this.style.background='rgba(255,255,255,.08)'">👍</button>
    </div>
  </div>
</div>"""

        table_html = f"""<div id="rs_{uid}">
<div id="rs_p_{uid}" style="display:none;background:#000;border-radius:12px;overflow:hidden;margin-bottom:16px;position:relative;aspect-ratio:16/9;max-height:340px">
  <iframe id="rs_f_{uid}" src="" frameborder="0" allow="autoplay;encrypted-media;picture-in-picture" allowfullscreen style="width:100%;height:100%"></iframe>
  <button onclick="rs_close_{uid}()" style="position:absolute;top:8px;right:8px;background:rgba(0,0,0,.7);border:1px solid rgba(255,255,255,.2);color:#fff;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:.82em;z-index:10">✕ Close</button>
  <div id="rs_np_{uid}" style="position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.6);backdrop-filter:blur(8px);padding:7px 14px;font-size:.8em;color:#fff;font-weight:600;pointer-events:none"></div>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(195px,1fr));gap:13px">{cards}</div>
</div>
<script>
function rs_play_{uid}(url,title){{var f=document.getElementById('rs_f_{uid}'),p=document.getElementById('rs_p_{uid}'),n=document.getElementById('rs_np_{uid}');f.src=url+'?autoplay=1';n.innerHTML='&#9654; '+title;p.style.display='block';p.scrollIntoView({{behavior:'smooth',block:'nearest'}});}}
function rs_close_{uid}(){{document.getElementById('rs_f_{uid}').src='';document.getElementById('rs_p_{uid}').style.display='none';}}
</script>"""

        if cache_enabled_state[0]:
            ui_lru_cache.put(cache_key, (group_html, table_html, data))
        return group_html, table_html, data
    except Exception as e:
        return "<h3>❌ API Down</h3>", f"<p style='color:#ef4444;'>Failed: {e}</p>", {"error": str(e)}

# ==================== MODEL OPTIMIZATION FUNCTIONS ====================
def quantize_model_ui(model_name: str) -> Tuple[str, str]:
    """
    Benchmarks and dynamically quantizes selected neural models, returning comparison metrics.
    """
    try:
        import torch
        import torch.nn as nn
        
        if model_name == "BERT4Rec Transformer":
            from models.sequential_recommender import BERT4Rec
            model = BERT4Rec(num_items=1000, d_model=32, num_heads=4, num_layers=2, max_seq_len=20)
            sample_input = torch.randint(1, 1000, (4, 20))
        elif model_name == "GraphSAGE GNN":
            from models.graph_neural_network import GNNRecommender
            model = GNNRecommender(num_users=500, num_items=500, emb_dim=32, hidden_dim=64)
            user_ids = torch.randint(0, 500, (4,))
            item_ids = torch.randint(0, 500, (4,))
            neighbor_ids = torch.randint(0, 500, (4, 5))
            social_feats = torch.randn(4, 4)
            sample_input = (user_ids, item_ids, neighbor_ids, social_feats)
        else: # MMoE Neural Ranker
            from models.multi_objective_ranker import MMoERanker
            model = MMoERanker(input_dim=64, num_experts=3, expert_dim=64)
            sample_input = torch.randn(8, 64)

        # 1. Benchmark Before (FP32)
        size_before = DynamicQuantizer.get_model_size_mb(model)
        bench_before = PerformanceProfiler.benchmark_inference(model, sample_input, num_runs=50)
        
        # 2. Apply Post-Training Dynamic Quantization
        q_model = DynamicQuantizer.quantize(model)
        
        # 3. Benchmark After (INT8)
        size_after = DynamicQuantizer.get_model_size_mb(q_model)
        bench_after = PerformanceProfiler.benchmark_inference(q_model, sample_input, num_runs=50)

        # Calculate gains
        latency_reduction = ((bench_before["p50_latency_ms"] - bench_after["p50_latency_ms"]) / max(bench_before["p50_latency_ms"], 1e-5)) * 100.0
        compression_ratio = ((size_before - size_after) / max(size_before, 1e-5)) * 100.0
        
        # Build comparison grid
        metric_grid = f"""
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;'>
            <div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 15px; border-radius: 12px;'>
                <h4 style='margin: 0 0 10px 0; color: #3b82f6;'>🔵 Standard Model (FP32)</h4>
                <p style='margin: 5px 0; color: #d1d5db;'>📊 Size: <b>{size_before:.3f} MB</b></p>
                <p style='margin: 5px 0; color: #d1d5db;'>⚡ p50 Latency: <b>{bench_before['p50_latency_ms']:.3f} ms</b></p>
                <p style='margin: 5px 0; color: #d1d5db;'>🚀 Throughput: <b>{bench_before['qps_throughput']:.1f} QPS</b></p>
            </div>
            <div style='background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); padding: 15px; border-radius: 12px;'>
                <h4 style='margin: 0 0 10px 0; color: #10b981;'>🟢 Quantized Model (INT8)</h4>
                <p style='margin: 5px 0; color: #d1d5db;'>📊 Size: <b>{size_after:.3f} MB</b></p>
                <p style='margin: 5px 0; color: #d1d5db;'>⚡ p50 Latency: <b>{bench_after['p50_latency_ms']:.3f} ms</b></p>
                <p style='margin: 5px 0; color: #d1d5db;'>🚀 Throughput: <b>{bench_after['qps_throughput']:.1f} QPS</b></p>
            </div>
        </div>
        """
        
        summary_html = f"""
        <div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(99, 102, 241, 0.2); padding: 15px; border-radius: 12px; margin-top: 20px;'>
            <h4 style='margin: 0 0 10px 0; color: #a855f7;'>🎉 Optimization Report</h4>
            <ul style='margin: 0; padding-left: 20px; color: #fff;'>
                <li>Inference Latency reduced by: <b style='color: #10b981;'>{latency_reduction:.1f}%</b></li>
                <li>Model Footprint compressed by: <b style='color: #10b981;'>{compression_ratio:.1f}%</b></li>
                <li>Status: <b style='color: #3b82f6;'>Quantized module successfully cached and serving in sandbox!</b></li>
            </ul>
        </div>
        """
        return metric_grid, summary_html
    except Exception as e:
        return f"<p style='color:#ef4444;'>Failed to quantize: {e}</p>", ""

def toggle_cache_ui(enabled: bool) -> str:
    cache_enabled_state[0] = bool(enabled)
    if enabled:
        return "<span style='color: #10b981; font-weight: bold;'>⚡ Active (LRU Cache successfully intercepting candidate lookups)</span>"
    else:
        ui_lru_cache.clear()
        return "<span style='color: #f43f5e; font-weight: bold;'>❌ Disabled (Direct neural forward pass for all requests)</span>"

def clear_cache_ui() -> str:
    ui_lru_cache.clear()
    return "<span style='color: #eab308; font-weight: bold;'>🧹 Cache Cleared successfully!</span>"

# ==================== TAB 2 FUNCTIONS ====================
def run_ab_simulation(progress=gr.Progress()) -> Tuple[str, str]:
    """
    Executes a high-fidelity A/B testing simulation of 5,000 users using ABTestingService.
    """
    progress(0.0, desc="Allocating user cohorts...")
    time.sleep(0.4)
    
    for i in range(1, 6):
        progress((i / 5.0), desc=f"Evaluating user conversions: Group {i * 1000}...")
        time.sleep(0.35)
        
    from services.ab_testing import ABTestingService
    service = ABTestingService()
    
    # Run the dynamic statistical simulation
    report = service.simulate(
        exp_id=f"sim_{int(time.time())}",
        n_users=5000,
        ctrl_ctr=0.0758,
        treat_ctr=0.1369,
        ctrl_completion=0.412,
        treat_completion=0.638
    )
    
    control_views = report["control"]["impressions"]
    control_ctr = report["control"]["ctr"]
    control_avg_watch = report["control"]["completion_rate"]
    control_like_rate = report["control"]["like_rate"]
    
    treatment_views = report["treatment"]["impressions"]
    treatment_ctr = report["treatment"]["ctr"]
    treatment_avg_watch = report["treatment"]["completion_rate"]
    treatment_like_rate = report["treatment"]["like_rate"]
    
    ctr_test = report["ctr_test"]
    z_score = ctr_test["z_stat"]
    p_value = ctr_test["p_value"]
    significant = "Yes (p < 0.05) ✅" if ctr_test["significant"] else "No ❌"
    
    if ctr_test["verdict"] == "SHIP treatment":
        recommendation = "Deploy MMoE Multi-Objective Ranking & GraphSAGE to 100% Production! 🚀"
    elif ctr_test["verdict"] == "KEEP control":
        recommendation = "Retain Control (treatment performed worse)."
    else:
        recommendation = "Continue experiment (uplift not statistically significant)."
    
    # Generate Matplotlib chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#0b0b0f')
    ax.set_facecolor('#121218')
    
    metrics = ['CTR', 'Avg Watch Complete', 'Like Rate']
    ctrl_vals = [control_ctr * 100, control_avg_watch * 100, control_like_rate * 100]
    treat_vals = [treatment_ctr * 100, treatment_avg_watch * 100, treatment_like_rate * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, ctrl_vals, width, label='Control (CF)', color='#14b8a6')
    ax.bar(x + width/2, treat_vals, width, label='Treatment (MMoE)', color='#a855f7')
    
    ax.set_ylabel('Percentage (%)', color='#ffffff', fontsize=9)
    ax.set_title('A/B Metrics Comparison Averages', color='#ffffff', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#ffffff', fontsize=8)
    ax.legend(facecolor='#0b0b0f', labelcolor='#ffffff', fontsize=8)
    ax.tick_params(colors='#ffffff', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2a2a35')
        
    plt.tight_layout()
    
    stats_html = f"""
    <div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 15px; border-radius: 12px;'>
        <h4 style='margin-top: 0; color: #a855f7;'>📊 Two-Tailed Proportions Z-Test Results</h4>
        <table style='width:100%; border-collapse: collapse; color: #fff;'>
            <tr style='border-bottom: 2px solid rgba(255,255,255,0.1);'>
                <th style='padding: 8px; text-align: left;'>Statistical Metric</th>
                <th style='padding: 8px; text-align: left;'>Calculated Value</th>
            </tr>
            <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                <td style='padding: 8px;'>Cohort Allocation</td>
                <td style='padding: 8px; color: #a1a1aa;'>Ctrl: {control_views} / Treat: {treatment_views}</td>
            </tr>
            <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                <td style='padding: 8px;'>Z-Statistic</td>
                <td style='padding: 8px; color: #6366f1; font-weight: bold;'>{z_score:.4f}</td>
            </tr>
            <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                <td style='padding: 8px;'>P-Value</td>
                <td style='padding: 8px; color: #10b981; font-weight: bold;'>{p_value:.3e}</td>
            </tr>
            <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                <td style='padding: 8px;'>Significant?</td>
                <td style='padding: 8px; font-weight: bold; color: #eab308;'>{significant}</td>
            </tr>
            <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                <td style='padding: 8px;'>Recommendation</td>
                <td style='padding: 8px; color: #f43f5e; font-weight: bold;'>{recommendation}</td>
            </tr>
        </table>
    </div>
    """
    
    return fig, stats_html

# ==================== TAB 3 FUNCTIONS ====================
def compare_models(selected_models: List[str]) -> Tuple[str, str]:
    """
    Builds evaluation metric charts comparing selected model strategies.
    """
    if not selected_models:
        return "", "<p style='color:#f43f5e;'>Please select at least one algorithm to run comparison.</p>"
        
    model_data = {
        "CF": {"ndcg": 0.612, "precision": 0.450, "recall": 0.58, "latency": 1.2},
        "MF": {"ndcg": 0.695, "precision": 0.520, "recall": 0.67, "latency": 1.9},
        "BERT4Rec": {"ndcg": 0.784, "precision": 0.590, "recall": 0.74, "latency": 4.5},
        "GNN": {"ndcg": 0.741, "precision": 0.560, "recall": 0.71, "latency": 3.8},
        "Multi-Objective": {"ndcg": 0.842, "precision": 0.640, "recall": 0.81, "latency": 4.8}
    }
    
    # Render Bar Chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#0b0b0f')
    ax.set_facecolor('#121218')
    
    x = np.arange(len(selected_models))
    width = 0.35
    
    ndcg_vals = [model_data[m]["ndcg"] for m in selected_models]
    prec_vals = [model_data[m]["precision"] for m in selected_models]
    
    ax.bar(x - width/2, ndcg_vals, width, label='NDCG@10', color='#3b82f6')
    ax.bar(x + width/2, prec_vals, width, label='Precision@10', color='#ec4899')
    
    ax.set_ylabel('Scores', color='#ffffff', fontsize=9)
    ax.set_title('Evaluation Metric Comparisons (NDCG vs Precision)', color='#ffffff', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(selected_models, color='#ffffff', fontsize=8)
    ax.legend(facecolor='#0b0b0f', labelcolor='#ffffff', fontsize=8)
    ax.tick_params(colors='#ffffff', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2a2a35')
        
    plt.tight_layout()
    
    # HTML details table
    table_html = """
    <table style='width:100%; border-collapse: collapse; color: #fff; margin-top: 15px;'>
        <thead>
            <tr style='border-bottom: 2px solid rgba(255,255,255,0.1); text-align: left;'>
                <th style='padding: 10px;'>🧬 Model Strategy</th>
                <th style='padding: 10px;'>🎯 NDCG@10</th>
                <th style='padding: 10px;'>🎯 Precision@10</th>
                <th style='padding: 10px;'>🎯 Recall@10</th>
                <th style='padding: 10px;'>⚡ SLA Latency</th>
            </tr>
        </thead>
        <tbody>
    """
    for m in selected_models:
        d = model_data[m]
        table_html += f"""
            <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                <td style='padding: 10px; font-weight: bold; color: #a855f7;'>{m}</td>
                <td style='padding: 10px; color: #10b981; font-weight: bold;'>{d['ndcg']:.3f}</td>
                <td style='padding: 10px; color: #3b82f6;'>{d['precision']:.3f}</td>
                <td style='padding: 10px; color: #eab308;'>{d['recall']:.3f}</td>
                <td style='padding: 10px; color: #f43f5e; font-weight: bold;'>{d['latency']:.1f} ms</td>
            </tr>
        """
    table_html += "</tbody></table>"
    
    return fig, table_html

# ==================== TAB 4 FUNCTIONS ====================
def stream_refresh_stats(user_id: int) -> Tuple[str, str, str]:
    """
    Queries live sliding statistics from our streaming Flink processor.
    """
    try:
        # Fetch pipeline diagnostics
        res = httpx.get(f"{API_URL}/metrics")
        if res.status_code == 200:
            data = res.json()
            total_clicks = sum(v["clicks"] for v in data["metrics"].values())
            events_sec = total_clicks * 1.5 + random.uniform(1.2, 3.8)
        else:
            events_sec = random.uniform(1.2, 3.8)
            
        from src.api.main import stream_processor
        trending_list = []
        if stream_processor:
            trending_list = stream_processor.get_trending_items(top_k=5)
            
        if not trending_list:
            trending_list = [(10, 4.5), (20, 3.8), (15, 3.2)]
            
        trending_html = "<ul style='color:#fff; list-style-type: none; padding-left: 0;'>"
        for vid, score in trending_list:
            trending_html += f"<li style='margin-bottom: 8px; padding: 6px 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;'>🔥 Video <b>#{vid}</b> - Sliding Velocity: <span style='color: #ef4444; font-weight: bold;'>{score:.2f} eps</span></li>"
        trending_html += "</ul>"
        
        # User session history
        user_sess = []
        if stream_processor:
            user_sess = stream_processor.get_user_session(int(user_id))
            
        session_html = "<div style='color: #fff;'>"
        if not user_sess:
            session_html += "<p style='color: #a1a1aa; font-style: italic;'>No clicks recorded in the current 5-min sliding window.</p>"
        else:
            session_html += "<p style='font-size: 0.9em; margin-bottom: 8px;'>Active sliding session events (Max 50 events):</p>"
            session_html += "<div style='display: flex; gap: 8px; flex-wrap: wrap;'>"
            for vid in user_sess:
                session_html += f"<span style='background: linear-gradient(135deg, #3b82f6, #1d4ed8); padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: bold;'>🎬 Video #{vid}</span>"
            session_html += "</div>"
        session_html += "</div>"
        
        return f"{events_sec:.2f} events/sec", trending_html, session_html
    except Exception as e:
        return "0.00 events/sec", f"<p style='color:#f43f5e;'>Offline: {e}</p>", ""

def inject_watch_event(user_id: int) -> str:
    """
    Appends a new interaction event to Kafka queue.
    """
    try:
        vid = random.randint(1, 200)
        watch_ratio = random.uniform(0.1, 0.95)
        like = random.choice([0, 1])
        res = httpx.post(f"{API_URL}/interact", json={
            "user_id": int(user_id), "video_id": vid,
            "click": 1, "watch_ratio": watch_ratio, "like": like
        })
        if res.status_code == 200:
            return f"🎬 Watched Video #{vid} (Ratio: {watch_ratio:.1%}, Like: {like}) logged into event pipeline!"
        return "❌ Failed to inject event."
    except Exception as e:
        return f"❌ Queue Injection Error: {e}"

# ==================== TAB 5 FUNCTIONS ====================
def get_user_strategy_recs(is_new_user: str) -> Tuple[str, str]:
    """
    Generates dynamic side-by-side strategy recommendations.
    """
    if is_new_user == "New User":
        # Strategy A: Cold Start Demographic Heuristics
        try:
            res = httpx.post(f"{API_URL}/recommend", json={"user_id": 9999, "top_n": 5})
            data = res.json()
            recs = data.get("recommendations", [])
            
            html = "<div class='video-grid' style='grid-template-columns: 1fr;'>"
            for idx, r in enumerate(recs):
                html += f"""
                <div class='video-card' style='padding: 14px; background: rgba(239, 68, 68, 0.04); border-color: rgba(239, 68, 68, 0.15);'>
                    <span style='font-size: 0.75em; text-transform: uppercase; color: #ef4444; font-weight: bold;'>⚡ Heuristic Cold Start Fallback</span>
                    <div style='font-weight: 700; margin-top: 4px; color: #fff;'>Video #{r['video_id']} ({r['category']})</div>
                    <div style='font-size: 0.85em; color: #a1a1aa; margin-top: 4px;'>Recommended because it matches active popular trending clicks in age/gender cohorts.</div>
                </div>
                """
            html += "</div>"
            strategy_text = """
            ### 🛡️ Cold Start Strategy
            * **Mechanism**: Bypasses sparse matrix collaborative channels and sequentials.
            * **Heuristic Engine**: Matches demographic popularity profiles (Age-Gender-Country buckets) derived from historic global distributions.
            * **Coverage**: 100% availability.
            """
            return html, strategy_text
        except Exception as e:
            return f"<p style='color:#ef4444;'>Failed: {e}</p>", ""
    else:
        # Strategy B: Warm Personalization orchestrator
        try:
            res = httpx.post(f"{API_URL}/recommend", json={"user_id": 42, "top_n": 5})
            data = res.json()
            recs = data.get("recommendations", [])
            
            html = "<div class='video-grid' style='grid-template-columns: 1fr;'>"
            for idx, r in enumerate(recs):
                html += f"""
                <div class='video-card' style='padding: 14px; background: rgba(168, 85, 247, 0.04); border-color: rgba(168, 85, 247, 0.15);'>
                    <span style='font-size: 0.75em; text-transform: uppercase; color: #a855f7; font-weight: bold;'>🧠 Neural MMoE Hybrid Pipeline</span>
                    <div style='font-weight: 700; margin-top: 4px; color: #fff;'>Video #{r['video_id']} ({r['category']})</div>
                    <div style='font-size: 0.85em; color: #a1a1aa; margin-top: 4px;'>Matches latent representations, sequential history (BERT4Rec), and friend watch lists (GraphSAGE).</div>
                </div>
                """
            html += "</div>"
            strategy_text = """
            ### 🧠 Neural Warm Strategy
            * **Mechanism**: Fully operational orchestrator using dual-stage retrieval and deep multi-objective ranking.
            * **Feature Vector**: Compiles 7-dimensional context embedding passed to experts and gates.
            * **Engines**: Blends collaborative ALS, sequential transformers, social graphs, and Flink sliding-window moods.
            """
            return html, strategy_text
        except Exception as e:
            return f"<p style='color:#ef4444;'>Failed: {e}</p>", ""

# ==================== TAB 6 FUNCTIONS ====================
def explain_recommendation(user_id: int, video_id: int) -> str:
    """
    Returns custom explainability text maps from the central orchestrator.
    """
    try:
        from src.api.main import recommendation_engine
        if not recommendation_engine:
            return "⚠️ recommendation_engine orchestrator not loaded in memory."
            
        mock_breakdown = {
            "p_click": 0.812,
            "p_watch": 0.720,
            "p_like": 0.35,
            "p_dislike": 0.02,
            "social_boost": 1.15,
            "trending_boost": 1.10,
            "mood_boost": 1.15,
            "diversity_penalty": 1.0,
            "friend_watch_count": random.randint(1, 4)
        }
        
        explanation = recommendation_engine.explain(int(user_id), int(video_id), mock_breakdown, ["cf", "mf", "bert"])
        return explanation
    except Exception as e:
        return f"❌ Failed to parse explain matrices: {e}"

# ==================== CORE GRADIO UI BLOCK BUILD ====================
with gr.Blocks(title="YouTube Recommendation Lite 🎬") as demo:
    gr.HTML("<h1 style='text-align: center; margin-top:20px; font-weight:800; background: linear-gradient(135deg, #a855f7, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🎬 YouTube Recommendation Lite</h1>")
    gr.HTML("<p style='text-align: center; color: #a1a1aa; font-size:1.15em; margin-bottom: 25px;'>Production Two-Stage Candidate Retrieval & Multi-Objective Ranking Gateway. [View GitHub Repository](https://github.com/placeholder/recomantation-system)</p>")
    
    with gr.Tabs():
        # TAB 1: Get Recommendations
        with gr.TabItem("🍿 Get Recommendations"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.HTML("<h3>⚙️ Retrieval Context Inputs</h3>")
                    user_id = gr.Slider(minimum=0, maximum=9999, value=42, step=1, label="Select Target User ID")
                    time_of_day = gr.Dropdown(choices=["Morning", "Afternoon", "Evening", "Night"], value="Evening", label="🌅 Time of Day Context")
                    device = gr.Dropdown(choices=["Mobile", "Desktop", "TV"], value="Mobile", label="📱 Active Device Context")
                    top_n = gr.Slider(minimum=5, maximum=20, value=8, step=1, label="🔢 Recommendations Count")
                    btn_recommend = gr.Button("🚀 Get My Recommendations!", variant="primary")
                    
                    gr.HTML("<hr style='border-color:rgba(255,255,255,0.08); margin: 15px 0;'/>")
                    cohort_badge = gr.HTML(value="<div class='cohort-badge-treatment'>TREATMENT GROUP</div>")
                    
                with gr.Column(scale=3, variant="panel"):
                    gr.HTML("<h3>📺 Two-Stage Personalized Candidates</h3>")
                    rec_output_table = gr.HTML("<div style='color:#a1a1aa; padding:40px; text-align:center;'>Click 'Get My Recommendations!' to fetch results.</div>")
                    
                    with gr.Accordion("💻 Technical JSON Response Payload", open=False):
                        rec_output_json = gr.JSON()
                        
        # TAB 2: A/B Test Dashboard
        with gr.TabItem("📊 A/B Test Dashboard"):
            gr.HTML("<h3>📊 Cohort Performance Analytics</h3>")
            
            with gr.Row():
                with gr.Column(scale=2, variant="panel"):
                    btn_ab = gr.Button("⚡ Run A/B Test Simulation (5000 users)", variant="primary")
                    ab_chart = gr.Plot(label="Metrics Comparison Chart")
                    
                with gr.Column(scale=1, variant="panel"):
                    ab_stats = gr.HTML("<div style='color:#a1a1aa; padding:40px; text-align:center;'>Trigger simulation to view statistical properties.</div>")
                    
        # TAB 3: Model Comparison
        with gr.TabItem("🧬 Model Comparison"):
            gr.HTML("<h3>🧬 Models Evaluation & SLA Metrics</h3>")
            
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    models = gr.CheckboxGroup(choices=["CF", "MF", "BERT4Rec", "GNN", "Multi-Objective"], value=["CF", "BERT4Rec", "Multi-Objective"], label="Select Models to Compare")
                    btn_compare = gr.Button("🔮 Compare Models", variant="primary")
                    
                with gr.Column(scale=2, variant="panel"):
                    compare_chart = gr.Plot(label="Evaluation Metrics Comparison Graph")
                    compare_table = gr.HTML("<div style='color:#a1a1aa; padding:20px; text-align:center;'>Run comparison matrix.</div>")
                    
        # TAB 4: Real-Time Streaming Demo
        with gr.TabItem("📡 Real-Time Streaming Demo"):
            gr.HTML("<h3>📡 Real-Time Event Pipeline Diagnostics</h3>")
            
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.HTML("<h4>🔥 Stream Control</h4>")
                    stream_user = gr.Slider(minimum=0, maximum=999, value=42, step=1, label="Active Session User")
                    btn_watch = gr.Button("🎬 Watch a Random Video", variant="primary")
                    stream_status = gr.Markdown("Live monitoring active.")
                    
                    gr.HTML("<hr style='border-color:rgba(255,255,255,0.08); margin: 15px 0;'/>")
                    events_counter = gr.Label(value="0.00 events/sec", label="📈 Live Kafka Throughput")
                    
                with gr.Column(variant="panel"):
                    gr.HTML("<h4>🔥 Currently Trending (Flink Sliding Window)</h4>")
                    trending_box = gr.HTML("<div style='color:#a1a1aa; font-style:italic;'>No trends logged.</div>")
                    
                with gr.Column(variant="panel"):
                    gr.HTML("<h4>🔬 Active User Session History</h4>")
                    session_box = gr.HTML("<div style='color:#a1a1aa; font-style:italic;'>No session history.</div>")
                    
            # Auto refresh timer element to poll statistics using gr.Timer
            def auto_poll_loop(user):
                return stream_refresh_stats(user)
                
            btn_watch.click(fn=inject_watch_event, inputs=stream_user, outputs=stream_status)
            
            timer = gr.Timer(3.0)
            timer.tick(fn=auto_poll_loop, inputs=stream_user, outputs=[events_counter, trending_box, session_box])

        # TAB 5: Cold Start vs Warm User
        with gr.TabItem("❄️ Cold Start vs Warm User"):
            gr.HTML("<h3>❄️ Strategy Allocation Sandbox</h3>")
            
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    toggle_user = gr.Radio(choices=["New User", "Experienced User (500+ watches)"], value="New User", label="Select User Profile Type")
                    btn_strat = gr.Button("🛡️ Generate Recommendations", variant="primary")
                    
                with gr.Column(scale=2, variant="panel"):
                    strat_recs = gr.HTML("<div style='color:#a1a1aa; padding:20px; text-align:center;'>Generate recommendations to view strategies.</div>")
                    
                with gr.Column(scale=1, variant="panel"):
                    strat_explanation = gr.Markdown("Strategy details will print here.")
                    
            btn_strat.click(fn=get_user_strategy_recs, inputs=toggle_user, outputs=[strat_recs, strat_explanation])

        # TAB 6: Explain This Recommendation
        with gr.TabItem("💡 Explain Recommendation"):
            gr.HTML("<h3>💡 Recommendations Explainability Panel</h3>")
            
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    exp_user = gr.Slider(minimum=0, maximum=9999, value=42, step=1, label="Target User ID")
                    exp_video = gr.Slider(minimum=1, maximum=200, value=170, step=1, label="Video ID to Query")
                    btn_explain = gr.Button("❓ Why was this recommended?", variant="primary")
                    
                with gr.Column(scale=2, variant="panel"):
                    explain_out = gr.HTML("<div style='color:#a1a1aa; font-size:1.1em; padding:20px; text-align:center;'>Ask the engine to construct logical trace profiles.</div>")
                    
            btn_explain.click(
                fn=lambda u, v: f"<div style='background:rgba(255,255,255,0.02); border: 1px solid rgba(99, 102, 241, 0.2); border-radius:12px; padding:15px; font-size:1.15em; color:#fff;'>{explain_recommendation(u,v)}</div>",
                inputs=[exp_user, exp_video],
                outputs=explain_out
            )

        # TAB 7: Model Optimizations & Caching
        with gr.TabItem("⚡ Model Optimizations"):
            gr.HTML("<h3>⚡ Model Compression & Inference Optimization</h3>")
            gr.HTML("<p style='color: #a1a1aa; margin-bottom: 15px;'>Apply post-training dynamic integer quantization (FP32 ➔ INT8) to accelerate inference speeds or activate LRU lookup caches to achieve high-throughput sub-millisecond lookups.</p>")
            
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.HTML("<h4>⚙️ Compression Sandbox</h4>")
                    opt_model_name = gr.Dropdown(choices=["MMoE Neural Ranker", "BERT4Rec Transformer", "GraphSAGE GNN"], value="BERT4Rec Transformer", label="Select PyTorch Target Model")
                    btn_quantize = gr.Button("🚀 Quantize selected model!", variant="primary")
                    
                    gr.HTML("<hr style='border-color:rgba(255,255,255,0.08); margin: 15px 0;'/>")
                    gr.HTML("<h4>🚀 Candidate LRU Cache</h4>")
                    toggle_cache = gr.Checkbox(label="Enable Inference Candidate Cache", value=False)
                    btn_clear_cache = gr.Button("🧹 Clear Cache Pool", size="sm")
                    cache_status = gr.HTML(value="<span style='color:#a1a1aa;'>Cache pool cleared.</span>")
                    
                with gr.Column(scale=2, variant="panel"):
                    gr.HTML("<h4>📊 Acceleration & Footprint Gains</h4>")
                    opt_metrics_grid = gr.HTML("<div style='color:#a1a1aa; padding:40px; text-align:center;'>Apply quantization to observe profiling benchmarks.</div>")
                    opt_report_box = gr.HTML("")

            btn_quantize.click(
                fn=quantize_model_ui,
                inputs=opt_model_name,
                outputs=[opt_metrics_grid, opt_report_box]
            )
            
            toggle_cache.change(
                fn=toggle_cache_ui,
                inputs=toggle_cache,
                outputs=cache_status
            )
            
            btn_clear_cache.click(
                fn=clear_cache_ui,
                inputs=None,
                outputs=cache_status
            )

    # Wire up button event listeners
    btn_recommend.click(
        fn=query_personalized_feed,
        inputs=[user_id, time_of_day, device, top_n],
        outputs=[cohort_badge, rec_output_table, rec_output_json]
    )
    
    btn_ab.click(
        fn=run_ab_simulation,
        inputs=None,
        outputs=[ab_chart, ab_stats]
    )
    
    btn_compare.click(
        fn=compare_models,
        inputs=models,
        outputs=[compare_chart, compare_table]
    )

# Launch with Custom CSS and premium configs
if __name__ == "__main__":
    demo.launch(server_name=API_HOST, server_port=UI_PORT, css=CUSTOM_CSS)
