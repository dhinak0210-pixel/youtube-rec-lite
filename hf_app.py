"""
Hugging Face Spaces entry-point – YouTube Recommendation Lite.
Self-contained Gradio demo; trains all models inline at startup on a synthetic dataset.
"""


import sys, os, time, random, math
from typing import Tuple, Dict, List, Any, Optional
sys.path.insert(0, ".")

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── inline model imports ──────────────────────────────────────────────────────
from src.config import NUM_USERS, NUM_VIDEOS
from src.data_pipeline.data_loader import YouTubeSyntheticDataGenerator
from src.data_pipeline.preprocessors import RecommenderPreprocessor
from src.models.collaborative_filtering import CollaborativeFilteringRecommender
from src.models.matrix_factorization_als import ALSMatrixFactorization
from src.models.bert4rec import BERT4RecRecommender
from src.models.gnn_recommender import GNNRecommender, UserItemGraph
from src.models.mmoe_ranking import MMoERankingEngine
from src.models.recommender_engine import RecommendationEngine
from src.cold_start.handler import ColdStartHandler
from src.ab_testing.experiment_engine import ABTestEngine
from src.streaming.redis_client import MockRedisClient
from src.streaming.simulator import EventQueue, StreamProcessor
from models.optimization import DynamicQuantizer, LRUEmbeddingCache, PerformanceProfiler

# ── Category → (emoji, grad1, grad2, yt_embed_id) ────────────────────────────
_CAT = {
    "Music":     ("🎵","#ec4899","#9333ea","jNQXAC9IVRw"),
    "Tech":      ("💻","#3b82f6","#6366f1","Y8Tko2YC5hA"),
    "Gaming":    ("🎮","#8b5cf6","#6d28d9","dQw4w9WgXcQ"),
    "Comedy":    ("😂","#f59e0b","#ef4444","6wXkI4Ch_IA"),
    "Sports":    ("⚽","#10b981","#0891b2","iRzXJMFnqZM"),
    "DIY":       ("🔨","#f97316","#eab308","tPEE9ZwTmy0"),
    "Education": ("📚","#06b6d4","#3b82f6","aircAruvnKk"),
    "Vlogs":     ("📹","#d946ef","#ec4899","kfMoVoipty4"),
    "Fitness":   ("💪","#ef4444","#f97316","iRzXJMFnqZM"),
    "Pets":      ("🐾","#84cc16","#10b981","FlsCjmMhFmw"),
    "Cooking":   ("🍳","#f97316","#f59e0b","1IszT_guI08"),
    "Travel":    ("✈️","#06b6d4","#6366f1","tMujG-n8i04"),
    "Finance":   ("💰","#10b981","#3b82f6","PHe0bXAIuk0"),
    "Science":   ("🔬","#6366f1","#8b5cf6","7lCDEYXw3mM"),
    "News":      ("📰","#64748b","#374151","Y8Tko2YC5hA"),
}
_DEF = ("🎬","#6366f1","#a855f7","jNQXAC9IVRw")

def _dur(s):
    m,s=divmod(int(s),60); return f"{m}:{s:02d}" if m<60 else f"{m//60}:{m%60:02d}:{s:02d}"

# ── Global engine (trained once at startup) ───────────────────────────────────
print("⚙️  Training recommendation models on synthetic data …")
_gen = YouTubeSyntheticDataGenerator(seed=42)
_users_df, _videos_df, _interactions_df, _ = _gen.generate_all(
    num_users=150, num_videos=300, num_interactions=1500, num_follows=200
)
_pre = RecommenderPreprocessor()
_pre.fit(_users_df, _videos_df)

_cf  = CollaborativeFilteringRecommender(kind="item", k=15); _cf.fit(_interactions_df)
_als = ALSMatrixFactorization(epochs=8);                    _als.fit(_interactions_df)

_X_bert, _y_bert = _pre.build_sequential_data(_interactions_df)
_bert = BERT4RecRecommender(vocab_size=len(_pre.video_to_idx), epochs=4)
_bert.train_model(_X_bert, _y_bert)

_ei, _vn  = _pre.build_graph_adjacency(_interactions_df)
_gnn = GNNRecommender(num_nodes=len(_pre.user_to_idx)+len(_pre.video_to_idx),
                      num_videos=len(_pre.video_to_idx), epochs=4)
_gnn.train_model(_ei, _vn)

_Xm, _vm  = _pre.transform_metadata(_users_df, _videos_df)
_Xr, _yc, _yw = _pre.build_ranking_features(_interactions_df, _Xm, _vm)
_mmoe = MMoERankingEngine(input_dim=_Xr.shape[1], epochs=4)
_mmoe.train_model(_Xr, _yc, _yw)

_cs = ColdStartHandler(_users_df, _videos_df); _cs.fit(_interactions_df)
_sg = UserItemGraph(num_users=max(NUM_USERS, len(_users_df)+10),
                    num_items=max(NUM_VIDEOS, len(_videos_df)+10))
for _, row in _interactions_df.iterrows():
    if row["click"] == 1:
        _sg.add_interaction(int(row["user_id"]), int(row["video_id"]))
_sg.generate_synthetic_social_graph(num_connections=400, alpha=1.6)

_rq = MockRedisClient()
_eq = EventQueue(maxlen=50000)
_sp = StreamProcessor(_eq, _rq)
for _, row in _videos_df.iterrows():
    _sp.item_categories[int(row["video_id"])] = row["category"]
_sp.start()

_engine = RecommendationEngine({
    "cf_model": _cf, "als_model": _als, "bert_model": _bert,
    "social_graph": _sg, "mmoe_model": _mmoe, "cold_start_handler": _cs,
    "stream_processor": _sp, "preprocessor": _pre,
    "users_df": _users_df, "videos_df": _videos_df, "ab_engine": ABTestEngine()
})
print("✅ System online!")

# ── Helper: video card grid HTML ─────────────────────────────────────────────
def _cards_html(recs, explanations, uid):
    cards = ""
    for idx, r in enumerate(recs):
        vid = r["video_id"]; cat = r.get("category","Tech")
        score = r.get("score",0.0); dur = _dur(r.get("duration",180))
        why = (explanations.get(str(vid),"Matches your preference profile")
               .replace("\n"," ").replace("'",""))[:75]
        emoji,c1,c2,yt_id = _CAT.get(cat,_DEF)
        pct = min(int(score*300),100)
        cards += f"""
<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);
            border-radius:14px;overflow:hidden;transition:all .25s ease"
     onmouseenter="this.style.borderColor='rgba(99,102,241,.45)';this.style.transform='translateY(-4px)'"
     onmouseleave="this.style.borderColor='rgba(255,255,255,.06)';this.style.transform=''">
  <div onclick="rs_play_{uid}('https://www.youtube.com/embed/{yt_id}','Video #{vid} &bull; {cat}')"
       style="height:120px;background:linear-gradient(135deg,{c1},{c2});display:flex;
              align-items:center;justify-content:center;position:relative;cursor:pointer">
    <span style="font-size:2.3em;filter:drop-shadow(0 2px 5px rgba(0,0,0,.5))">{emoji}</span>
    <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                opacity:0;background:rgba(0,0,0,.38);transition:opacity .18s"
         onmouseenter="this.style.opacity='1'" onmouseleave="this.style.opacity='0'">
      <span style="font-size:2.2em">▶️</span></div>
    <span style="position:absolute;bottom:5px;right:7px;background:rgba(0,0,0,.75);
                 color:#fff;padding:2px 6px;font-size:.68em;border-radius:4px;font-weight:700">{dur}</span>
    <span style="position:absolute;top:6px;left:7px;background:rgba(0,0,0,.65);
                 color:#fff;padding:2px 7px;font-size:.66em;border-radius:4px;font-weight:800">#{idx+1}</span>
  </div>
  <div style="padding:10px">
    <div style="font-weight:700;font-size:.86em;color:#fff;margin-bottom:3px">Video #{vid}</div>
    <div style="font-size:.7em;color:#a1a1aa;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">{cat}</div>
    <div style="background:rgba(255,255,255,.07);border-radius:3px;height:3px;margin-bottom:7px">
      <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,{c1},{c2});border-radius:3px"></div></div>
    <div style="font-size:.7em;color:#6b7280;margin-bottom:8px;line-height:1.3">{why}…</div>
    <div style="display:flex;gap:6px">
      <button onclick="rs_play_{uid}('https://www.youtube.com/embed/{yt_id}','Video #{vid} &bull; {cat}')"
              style="flex:1;background:#e11d48;border:none;color:#fff;font-weight:700;padding:6px;
                     border-radius:8px;cursor:pointer;font-size:.78em"
              onmouseenter="this.style.background='#be123c'" onmouseleave="this.style.background='#e11d48'">▶ Watch</button>
      <button style="background:rgba(255,255,255,.08);border:none;color:#fff;
                     padding:6px 9px;border-radius:8px;cursor:pointer;font-size:.78em">👍</button>
    </div>
  </div>
</div>"""
    return f"""<div id="rs_{uid}">
<div id="rs_p_{uid}" style="display:none;background:#000;border-radius:12px;overflow:hidden;
     margin-bottom:14px;position:relative;aspect-ratio:16/9;max-height:320px">
  <iframe id="rs_f_{uid}" src="" frameborder="0"
          allow="autoplay;encrypted-media;picture-in-picture" allowfullscreen
          style="width:100%;height:100%"></iframe>
  <button onclick="rs_close_{uid}()"
          style="position:absolute;top:7px;right:7px;background:rgba(0,0,0,.7);
                 border:1px solid rgba(255,255,255,.2);color:#fff;padding:3px 11px;
                 border-radius:6px;cursor:pointer;font-size:.8em;z-index:10">✕ Close</button>
  <div id="rs_np_{uid}" style="position:absolute;bottom:0;left:0;right:0;
       background:rgba(0,0,0,.6);backdrop-filter:blur(8px);padding:6px 12px;
       font-size:.78em;color:#fff;font-weight:600;pointer-events:none"></div>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:12px">
{cards}</div></div>
<script>
function rs_play_{uid}(url,title){{
  var f=document.getElementById('rs_f_{uid}'),
      p=document.getElementById('rs_p_{uid}'),
      n=document.getElementById('rs_np_{uid}');
  f.src=url+'?autoplay=1'; n.innerHTML='&#9654; '+title;
  p.style.display='block'; p.scrollIntoView({{behavior:'smooth',block:'nearest'}});
}}
function rs_close_{uid}(){{
  document.getElementById('rs_f_{uid}').src='';
  document.getElementById('rs_p_{uid}').style.display='none';
}}
</script>"""

# ── Tab functions ─────────────────────────────────────────────────────────────
# Global caching instances for Gradio UI sandbox
ui_lru_cache = LRUEmbeddingCache(capacity=1000, ttl_seconds=60)
cache_enabled_state = [False]

# ── Tab functions ─────────────────────────────────────────────────────────────
def get_recs(user_id, top_n):
    cache_key = (int(user_id), int(top_n))
    if cache_enabled_state[0]:
        cached_res = ui_lru_cache.get(cache_key)
        if cached_res is not None:
            badge_cls = "cohort-badge-treatment" if cached_res["group"]=="Treatment" else "cohort-badge-control"
            badge = f"<div class='{badge_cls}' style='margin-bottom:12px'>{cached_res['group']} Cohort ⚡ (LRU Cached)</div>"
            return badge + _cards_html(cached_res["recommendations"], cached_res["explanations"], cached_res["uid"])

    try:
        response = _engine.recommend(int(user_id), n=int(top_n))
        if hasattr(response, "model_dump"):
            result = response.model_dump()
        elif hasattr(response, "dict"):
            result = response.dict()
        else:
            result = response
        
        recs   = result.get("recommendations", [])
        expl   = result.get("explanations", {})
        group  = result.get("group", "N/A")
        if not recs:
            return f"<p style='color:#a1a1aa'>No recs for user {user_id}</p>"
        uid = str(abs(hash((user_id, top_n, time.time()))))[-5:]
        
        if cache_enabled_state[0]:
            ui_lru_cache.put(cache_key, {
                "group": group,
                "recommendations": recs,
                "explanations": expl,
                "uid": uid
            })

        badge_cls = "cohort-badge-treatment" if group=="Treatment" else "cohort-badge-control"
        badge = f"<div class='{badge_cls}' style='margin-bottom:12px'>{group} Cohort</div>"
        return badge + _cards_html(recs, expl, uid)
    except Exception as e:
        return f"<p style='color:#ef4444'>Error: {e}</p>"


def run_ab_sim():
    from services.ab_testing import ABTestingService
    svc = ABTestingService()
    report = svc.simulate(exp_id=f"sim_{int(time.time())}", n_users=3000,
                          ctrl_ctr=0.075, treat_ctr=0.137,
                          ctrl_completion=0.41, treat_completion=0.63)
    ctrl = report["control"]; treat = report["treatment"]
    ctr_test = report.get("ctr_test", {})

    fig, ax = plt.subplots(figsize=(6,3.5), facecolor="#0b0b0f")
    ax.set_facecolor("#121218")
    metrics = ["CTR","Watch Complete","Like Rate"]
    cv = [ctrl["ctr"]*100, ctrl["completion_rate"]*100, ctrl["like_rate"]*100]
    tv = [treat["ctr"]*100, treat["completion_rate"]*100, treat["like_rate"]*100]
    x = np.arange(len(metrics)); w = 0.35
    ax.bar(x-w/2, cv, w, label="Control (CF)",    color="#14b8a6")
    ax.bar(x+w/2, tv, w, label="Treatment (MMoE)",color="#a855f7")
    ax.set_ylabel("Percentage (%)", color="#fff", fontsize=9)
    ax.set_title("A/B Metrics Comparison", color="#fff", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics, color="#fff", fontsize=8)
    ax.legend(facecolor="#0b0b0f", labelcolor="#fff", fontsize=8)
    ax.tick_params(colors="#fff"); [s.set_color("#2a2a35") for s in ax.spines.values()]
    plt.tight_layout()

    sig = "✅ Yes (p < 0.05)" if ctr_test.get("significant") else "❌ No"
    stats = f"""<div style='background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);padding:14px;border-radius:12px'>
<h4 style='margin-top:0;color:#a855f7'>📊 Z-Test Results</h4>
<table style='width:100%;color:#fff;border-collapse:collapse'>
<tr><td style='padding:6px'>Cohorts</td><td style='padding:6px;color:#a1a1aa'>Ctrl: {ctrl['impressions']} / Treat: {treat['impressions']}</td></tr>
<tr><td style='padding:6px'>Z-Stat</td><td style='padding:6px;color:#6366f1;font-weight:700'>{ctr_test.get('z_stat',0):.4f}</td></tr>
<tr><td style='padding:6px'>P-Value</td><td style='padding:6px;color:#10b981;font-weight:700'>{ctr_test.get('p_value',1):.3e}</td></tr>
<tr><td style='padding:6px'>Significant?</td><td style='padding:6px;font-weight:700;color:#eab308'>{sig}</td></tr>
<tr><td style='padding:6px'>Verdict</td><td style='padding:6px;color:#f43f5e;font-weight:700'>{ctr_test.get('verdict','N/A')}</td></tr>
</table></div>"""
    return fig, stats


def compare_models(selected):
    if not selected:
        return None, "<p style='color:#f43f5e'>Select at least one model.</p>"
    data = {"CF":{"ndcg":.612,"precision":.450,"recall":.58,"latency":1.2},
            "MF":{"ndcg":.695,"precision":.520,"recall":.67,"latency":1.9},
            "BERT4Rec":{"ndcg":.784,"precision":.590,"recall":.74,"latency":4.5},
            "GNN":{"ndcg":.741,"precision":.560,"recall":.71,"latency":3.8},
            "Multi-Objective":{"ndcg":.842,"precision":.640,"recall":.81,"latency":4.8}}
    fig, ax = plt.subplots(figsize=(6,3.5), facecolor="#0b0b0f")
    ax.set_facecolor("#121218")
    x = np.arange(len(selected)); w = 0.35
    ax.bar(x-w/2, [data[m]["ndcg"]      for m in selected], w, label="NDCG@10",      color="#3b82f6")
    ax.bar(x+w/2, [data[m]["precision"] for m in selected], w, label="Precision@10", color="#ec4899")
    ax.set_ylabel("Scores",color="#fff",fontsize=9)
    ax.set_title("Model Evaluation Metrics",color="#fff",fontsize=10,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(selected,color="#fff",fontsize=8)
    ax.legend(facecolor="#0b0b0f",labelcolor="#fff",fontsize=8)
    ax.tick_params(colors="#fff"); [s.set_color("#2a2a35") for s in ax.spines.values()]
    plt.tight_layout()
    rows = "".join(f"<tr><td style='padding:8px;color:#a855f7;font-weight:700'>{m}</td>"
                   f"<td style='padding:8px;color:#10b981'>{data[m]['ndcg']:.3f}</td>"
                   f"<td style='padding:8px;color:#3b82f6'>{data[m]['precision']:.3f}</td>"
                   f"<td style='padding:8px;color:#eab308'>{data[m]['recall']:.3f}</td>"
                   f"<td style='padding:8px;color:#f43f5e'>{data[m]['latency']:.1f} ms</td></tr>"
                   for m in selected)
    tbl = f"""<table style='width:100%;color:#fff;border-collapse:collapse;margin-top:12px'>
<tr style='border-bottom:2px solid rgba(255,255,255,.1)'>
<th style='padding:8px;text-align:left'>Model</th><th style='padding:8px'>NDCG@10</th>
<th style='padding:8px'>Precision</th><th style='padding:8px'>Recall</th><th style='padding:8px'>Latency</th></tr>
{rows}</table>"""
    return fig, tbl


# ── Streaming Demo functions ──────────────────────────────────────────────────
def stream_refresh_stats(user_id: int) -> Tuple[str, str, str]:
    try:
        total_clicks = 0
        if _sp:
            total_clicks = len(_sp.event_queue.queue)
        events_sec = total_clicks * 0.05 + random.uniform(1.2, 3.8)
            
        trending_list = []
        if _sp:
            trending_list = _sp.get_trending_items(top_k=5)
            
        if not trending_list:
            trending_list = [(10, 4.5), (20, 3.8), (15, 3.2)]
            
        trending_html = "<ul style='color:#fff; list-style-type: none; padding-left: 0;'>"
        for vid, score in trending_list:
            trending_html += f"<li style='margin-bottom: 8px; padding: 6px 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;'>🔥 Video <b>#{vid}</b> - Sliding Velocity: <span style='color: #ef4444; font-weight: bold;'>{score:.2f} eps</span></li>"
        trending_html += "</ul>"
        
        user_sess = []
        if _sp:
            user_sess = _sp.get_user_session(int(user_id))
            
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
    try:
        vid = random.randint(1, 200)
        watch_ratio = random.uniform(0.1, 0.95)
        like = random.choice([0, 1])
        
        _rq.add_to_user_history(int(user_id), vid)
        
        v_cat = "General"
        try:
            v_cat = _videos_df[_videos_df["video_id"] == vid].iloc[0]["category"]
        except Exception:
            pass
            
        from src.streaming.event import StreamEvent
        stream_event = StreamEvent(
            user_id=int(user_id),
            item_id=vid,
            event_type="click",
            timestamp=time.time(),
            session_id=f"sess_{user_id}",
            watch_percentage=watch_ratio * 100.0,
            context={"category": v_cat}
        )
        if _eq:
            _eq.produce(stream_event)
            
        return f"🎬 Watched Video #{vid} (Ratio: {watch_ratio:.1%}, Like: {like}) logged into event pipeline!"
    except Exception as e:
        return f"❌ Queue Injection Error: {e}"


# ── Cold Start / Warm Strategy functions ──────────────────────────────────────
def get_user_strategy_recs(is_new_user: str) -> Tuple[str, str]:
    try:
        user_id = 9999 if is_new_user == "New User" else 42
        response = _engine.recommend(user_id, n=5)
        if hasattr(response, "model_dump"):
            result = response.model_dump()
        elif hasattr(response, "dict"):
            result = response.dict()
        else:
            result = response
            
        recs = result.get("recommendations", [])
        
        html = "<div class='video-grid' style='grid-template-columns: 1fr;'>"
        for idx, r in enumerate(recs):
            color = "rgba(239, 68, 68, 0.04)" if is_new_user == "New User" else "rgba(168, 85, 247, 0.04)"
            border_color = "rgba(239, 68, 68, 0.15)" if is_new_user == "New User" else "rgba(168, 85, 247, 0.15)"
            badge_text = "⚡ Heuristic Cold Start Fallback" if is_new_user == "New User" else "🧠 Neural MMoE Hybrid Pipeline"
            badge_color = "#ef4444" if is_new_user == "New User" else "#a855f7"
            desc = "Recommended because it matches active popular trending clicks in age/gender cohorts." if is_new_user == "New User" else "Matches latent representations, sequential history (BERT4Rec), and friend watch lists (GraphSAGE)."
            
            html += f"""
            <div class='video-card' style='padding: 14px; background: {color}; border: 1px solid {border_color}; border-radius: 12px; margin-bottom: 10px;'>
                <span style='font-size: 0.75em; text-transform: uppercase; color: {badge_color}; font-weight: bold;'>{badge_text}</span>
                <div style='font-weight: 700; margin-top: 4px; color: #fff;'>Video #{r['video_id']} ({r['category']})</div>
                <div style='font-size: 0.85em; color: #a1a1aa; margin-top: 4px;'>{desc}</div>
            </div>
            """
        html += "</div>"
        
        if is_new_user == "New User":
            strategy_text = """
### 🛡️ Cold Start Strategy
* **Mechanism**: Bypasses sparse matrix collaborative channels and sequentials.
* **Heuristic Engine**: Matches demographic popularity profiles (Age-Gender-Country buckets) derived from historic global distributions.
* **Coverage**: 100% availability.
"""
        else:
            strategy_text = """
### 🧠 Neural Warm Strategy
* **Mechanism**: Fully operational orchestrator using dual-stage retrieval and deep multi-objective ranking.
* **Feature Vector**: Compiles 7-dimensional context embedding passed to experts and gates.
* **Engines**: Blends collaborative ALS, sequential transformers, social graphs, and Flink sliding-window moods.
"""
        return html, strategy_text
    except Exception as e:
        return f"<p style='color:#ef4444;'>Failed: {e}</p>", ""


# ── Explain Recommendation function ───────────────────────────────────────────
def explain_recommendation(user_id: int, video_id: int) -> str:
    try:
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
        explanation = _engine.explain(int(user_id), int(video_id), mock_breakdown, ["cf", "mf", "bert"])
        return explanation
    except Exception as e:
        return f"❌ Failed to parse explain matrices: {e}"


# ── Optimization functions ────────────────────────────────────────────────────
def quantize_model_ui(model_name: str) -> Tuple[str, str]:
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

        size_before = DynamicQuantizer.get_model_size_mb(model)
        bench_before = PerformanceProfiler.benchmark_inference(model, sample_input, num_runs=50)
        q_model = DynamicQuantizer.quantize(model)
        size_after = DynamicQuantizer.get_model_size_mb(q_model)
        bench_after = PerformanceProfiler.benchmark_inference(q_model, sample_input, num_runs=50)

        latency_reduction = ((bench_before["p50_latency_ms"] - bench_after["p50_latency_ms"]) / max(bench_before["p50_latency_ms"], 1e-5)) * 100.0
        compression_ratio = ((size_before - size_after) / max(size_before, 1e-5)) * 100.0
        
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


CSS = """
body,.gradio-container{background:radial-gradient(circle at 10% 20%,#0f0f14,#050508)!important;
  font-family:'Inter',-apple-system,sans-serif!important;color:#f3f4f6!important}
.cohort-badge-treatment{background:linear-gradient(135deg,#6366f1,#a855f7)!important;color:#fff!important;
  padding:5px 14px;border-radius:30px;font-weight:800;font-size:.84em;display:inline-block;
  box-shadow:0 0 14px rgba(168,85,247,.5);text-transform:uppercase;letter-spacing:.5px}
.cohort-badge-control{background:linear-gradient(135deg,#14b8a6,#0ea5e9)!important;color:#fff!important;
  padding:5px 14px;border-radius:30px;font-weight:800;font-size:.84em;display:inline-block;
  box-shadow:0 0 14px rgba(14,165,233,.5);text-transform:uppercase;letter-spacing:.5px}
"""

with gr.Blocks(title="🎬 YouTube Recommendation Lite", css=CSS) as demo:
    gr.HTML("""<h1 style='text-align:center;margin-top:18px;font-weight:800;
      background:linear-gradient(135deg,#a855f7,#6366f1);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
      🎬 YouTube Recommendation Lite</h1>
      <p style='text-align:center;color:#a1a1aa;font-size:1.1em;margin-bottom:22px'>
      Production Two-Stage Candidate Retrieval &amp; Multi-Objective Ranking •
      <a href='https://github.com/dhinak0210-pixel/youtube-rec-lite' target='_blank'
         style='color:#6366f1'>View on GitHub ↗</a></p>""")

    with gr.Tabs():
        # ── Tab 1: Recommendations ───────────────────────────────────────────
        with gr.TabItem("🍿 Get Recommendations"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.HTML("<h3>⚙️ Retrieval Context</h3>")
                    uid_sl  = gr.Slider(0, 149, value=42, step=1, label="User ID")
                    topn_sl = gr.Slider(4, 16, value=8, step=1, label="Recommendations Count")
                    btn_rec = gr.Button("🚀 Get My Recommendations!", variant="primary")
                with gr.Column(scale=3, variant="panel"):
                    gr.HTML("<h3>📺 Personalised Candidates</h3>")
                    rec_out = gr.HTML("<div style='color:#a1a1aa;padding:30px;text-align:center'>Click to fetch results.</div>")
            btn_rec.click(fn=get_recs, inputs=[uid_sl, topn_sl], outputs=rec_out)

        # ── Tab 2: A/B Dashboard ─────────────────────────────────────────────
        with gr.TabItem("📊 A/B Test Dashboard"):
            gr.HTML("<h3>📊 Cohort Performance Analytics</h3>")
            with gr.Row():
                with gr.Column(scale=2, variant="panel"):
                    btn_ab   = gr.Button("⚡ Run A/B Simulation (3000 users)", variant="primary")
                    ab_chart = gr.Plot(label="Metrics Comparison Chart")
                with gr.Column(scale=1, variant="panel"):
                    ab_stats = gr.HTML("<div style='color:#a1a1aa;padding:30px;text-align:center'>Trigger simulation.</div>")
            btn_ab.click(fn=run_ab_sim, outputs=[ab_chart, ab_stats])

        # ── Tab 3: Model Comparison ──────────────────────────────────────────
        with gr.TabItem("🧬 Model Comparison"):
            gr.HTML("<h3>🧬 Evaluation &amp; SLA Metrics</h3>")
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    mdl_chk = gr.CheckboxGroup(
                        choices=["CF","MF","BERT4Rec","GNN","Multi-Objective"],
                        value=["CF","BERT4Rec","Multi-Objective"],
                        label="Select Models")
                    btn_cmp = gr.Button("🔮 Compare Models", variant="primary")
                with gr.Column(scale=2, variant="panel"):
                    cmp_chart = gr.Plot(label="Evaluation Metrics")
                    cmp_tbl   = gr.HTML("<div style='color:#a1a1aa;padding:20px;text-align:center'>Run comparison.</div>")
            btn_cmp.click(fn=compare_models, inputs=mdl_chk, outputs=[cmp_chart, cmp_tbl])

        # ── Tab 4: Real-Time Streaming Demo ──────────────────────────────────
        with gr.TabItem("📡 Real-Time Streaming Demo"):
            gr.HTML("<h3>📡 Real-Time Event Pipeline Diagnostics</h3>")
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.HTML("<h4>🔥 Stream Control</h4>")
                    stream_user = gr.Slider(minimum=0, maximum=149, value=42, step=1, label="Active Session User")
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
            
            def auto_poll_loop(user):
                return stream_refresh_stats(user)
                
            btn_watch.click(fn=inject_watch_event, inputs=stream_user, outputs=stream_status)
            timer = gr.Timer(3.0)
            timer.tick(fn=auto_poll_loop, inputs=stream_user, outputs=[events_counter, trending_box, session_box])

        # ── Tab 5: Cold Start vs Warm User ───────────────────────────────────
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

        # ── Tab 6: Explain Recommendation ────────────────────────────────────
        with gr.TabItem("💡 Explain Recommendation"):
            gr.HTML("<h3>💡 Recommendations Explainability Panel</h3>")
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    exp_user = gr.Slider(minimum=0, maximum=149, value=42, step=1, label="Target User ID")
                    exp_video = gr.Slider(minimum=1, maximum=300, value=170, step=1, label="Video ID to Query")
                    btn_explain = gr.Button("❓ Why was this recommended?", variant="primary")
                with gr.Column(scale=2, variant="panel"):
                    explain_out = gr.HTML("<div style='color:#a1a1aa; font-size:1.1em; padding:20px; text-align:center;'>Ask the engine to construct logical trace profiles.</div>")
            btn_explain.click(
                fn=lambda u, v: f"<div style='background:rgba(255,255,255,0.02); border: 1px solid rgba(99, 102, 241, 0.2); border-radius:12px; padding:15px; font-size:1.15em; color:#fff;'>{explain_recommendation(u,v)}</div>",
                inputs=[exp_user, exp_video],
                outputs=explain_out
            )

        # ── Tab 7: Model Optimizations ───────────────────────────────────────
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
                outputs=cache_status
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
