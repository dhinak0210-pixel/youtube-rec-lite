import time
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple, Dict, Set
from src.streaming.redis_client import MockRedisClient
from src.utils.logger import logger
from src.config import NUM_USERS, NUM_VIDEOS

@dataclass
class StreamEvent:
    """
    Standard Real-Time Event schema representing a single user action.
    """
    user_id: int
    item_id: int
    event_type: str
    timestamp: float
    session_id: str
    watch_percentage: float
    context: dict = field(default_factory=dict)

class EventQueue:
    """
    Thread-safe Message Queue simulating Apache Kafka using a bounded double-ended queue.
    """
    def __init__(self, maxlen: int = 100000):
        self.queue = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.total_produced = 0
        self.total_consumed = 0
        self.start_time = time.time()
        
    def produce(self, event: StreamEvent):
        with self.lock:
            self.queue.append(event)
            self.total_produced += 1
            
    def consume_batch(self, batch_size: int = 100) -> List[StreamEvent]:
        batch = []
        with self.lock:
            for _ in range(min(batch_size, len(self.queue))):
                batch.append(self.queue.popleft())
                self.total_consumed += 1
        return batch
        
    def stats(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.start_time
            throughput = self.total_produced / elapsed if elapsed > 0 else 0.0
            return {
                "queue_size": len(self.queue),
                "lag": len(self.queue),
                "throughput_events_per_sec": throughput,
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed
            }

class StreamProcessor:
    """
    Real-Time Stream Processor simulating Apache Flink.
    Runs continuously in a background daemon thread to process batches,
    calculating slide window counts and mood aggregates.
    """
    def __init__(self, queue: EventQueue, redis_client: MockRedisClient):
        self.queue = queue
        self.redis_client = redis_client
        self.is_running = False
        self.lock = threading.Lock()
        self._thread = None
        
        # Flink Sliding Window Storage (in-memory state)
        self.sessions: Dict[int, List[Tuple[int, float, str]]] = {}  # user_id -> [(item_id, timestamp, category)]
        self.item_views: Dict[int, List[float]] = {}                  # item_id -> [timestamps]
        self.user_moods: Dict[int, str] = {}                          # user_id -> current mood category
        self.item_categories: Dict[int, str] = {}                     # item_id -> category cache
        
    def _process_loop(self):
        while self.is_running:
            batch = self.queue.consume_batch(batch_size=100)
            if batch:
                now = time.time()
                for event in batch:
                    # a) Update user session (last 50 actions, 5-min window = 300s)
                    self._update_session(event, now)
                    
                    # b) Update real-time item popularity (last 5 min = 300s)
                    self._update_item_popularity(event, now)
                    
                    # c) Update user's current mood (dominant category last 10 mins = 600s)
                    self._update_user_mood(event, now)
                    
                    # d) Sync dynamic telemetry back to fakeredis Feature Store
                    self._sync_to_feature_store(event)
                    
            time.sleep(0.01) # Sleep 10ms

    def _update_session(self, event: StreamEvent, now: float):
        with self.lock:
            user_id = event.user_id
            if user_id not in self.sessions:
                self.sessions[user_id] = []
            
            category = event.context.get("category", "General")
            self.sessions[user_id].append((event.item_id, event.timestamp, category))
            
            # Slide window clean: keep last 50 actions within 5 minutes (300 seconds)
            cutoff = now - 300.0
            self.sessions[user_id] = [
                act for act in self.sessions[user_id]
                if act[1] >= cutoff
            ][-50:]

    def _update_item_popularity(self, event: StreamEvent, now: float):
        with self.lock:
            item_id = event.item_id
            category = event.context.get("category", "General")
            self.item_categories[item_id] = category
            
            if item_id not in self.item_views:
                self.item_views[item_id] = []
            self.item_views[item_id].append(event.timestamp)
            
            # Slide window clean: keep clicks within 5 minutes (300 seconds)
            cutoff = now - 300.0
            self.item_views[item_id] = [
                t for t in self.item_views[item_id]
                if t >= cutoff
            ]

    def _update_user_mood(self, event: StreamEvent, now: float):
        with self.lock:
            user_id = event.user_id
            if user_id not in self.sessions:
                return
            
            # Mood window represents dominant category in last 10 minutes (600 seconds)
            cutoff = now - 600.0
            recent_cats = [
                act[2] for act in self.sessions[user_id]
                if act[1] >= cutoff
            ]
            
            if recent_cats:
                dominant_mood = max(set(recent_cats), key=recent_cats.count)
                self.user_moods[user_id] = dominant_mood

    def _sync_to_feature_store(self, event: StreamEvent):
        # Synchronize action queue directly into redis mock DB
        self.redis_client.add_to_user_history(event.user_id, event.item_id)

    def get_trending_items(self, top_k: int = 10) -> List[Tuple[int, int]]:
        """
        Returns top trending items based on click counts in the active 5-min window.
        """
        with self.lock:
            now = time.time()
            cutoff = now - 300.0
            counts = {}
            for item_id, timestamps in self.item_views.items():
                active_views = sum(1 for t in timestamps if t >= cutoff)
                if active_views > 0:
                    counts[item_id] = active_views
                    
            sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return sorted_items[:top_k]

    def get_user_session(self, user_id: int) -> List[int]:
        with self.lock:
            if user_id not in self.sessions:
                return []
            return [act[0] for act in self.sessions[user_id]]

    def get_user_current_mood(self, user_id: int) -> str:
        with self.lock:
            return self.user_moods.get(user_id, "General")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)

class SessionAwareRecommender:
    """
    Session-Aware Recommender wrapper that dynamically adjusts candidate scores:
    - Excludes last 10 recently watched items.
    - Boosts active real-time trending items (1.10x).
    - Boosts current user mood category match (1.15x).
    """
    def __init__(self, base_recommender, stream_processor: StreamProcessor):
        self.base_recommender = base_recommender
        self.stream_processor = stream_processor
        
    def recommend(self, user_id: int, top_n: int = 10) -> Dict[str, any]:
        # 1. Fetch base candidates
        try:
            base_candidates = self.base_recommender.retrieve_candidates(user_id, top_n=top_n * 3)
        except Exception:
            # Fallback mock items
            base_candidates = [(item, random.uniform(0.1, 0.9)) for item in random.sample(range(1, 5000), top_n * 3)]
            
        # 2. Fetch real-time streaming states from Flink simulator
        session_items = self.stream_processor.get_user_session(user_id)
        current_mood = self.stream_processor.get_user_current_mood(user_id)
        trending = dict(self.stream_processor.get_trending_items(top_k=20))
        
        # 3. Apply session adjustments
        recently_watched = set(session_items[-10:])
        boosted_candidates = []
        
        for item_id, score in base_candidates:
            if item_id in recently_watched:
                continue
                
            boosted_score = score
            
            # Boost trending items (1.10x)
            if item_id in trending:
                boosted_score *= 1.10
                
            # Boost current user mood matches (1.15x)
            item_cat = self.stream_processor.item_categories.get(item_id, "General")
            if item_cat == current_mood and current_mood != "General":
                boosted_score *= 1.15
                
            boosted_candidates.append((item_id, boosted_score))
            
        # Re-sort boosted candidates
        boosted_candidates.sort(key=lambda x: x[1], reverse=True)
        final_recs = boosted_candidates[:top_n]
        
        return {
            "user_id": user_id,
            "recommendations": final_recs,
            "session_length": len(session_items),
            "current_mood": current_mood,
            "num_trending_boosted": sum(1 for item, _ in final_recs if item in trending)
        }


# =====================================================================
# BACKWARD COMPATIBLE SYSTEM DESIGN SIMULATOR CODES
# =====================================================================

class RealTimeTrafficSimulator:
    """
    Simulates real-time click stream traffic from active users.
    """
    def __init__(self, click_callback: Optional[Callable[[int, int], None]] = None):
        self.redis_client = MockRedisClient()
        self.click_callback = click_callback
        self.is_running = False
        self._thread = None

    def _simulation_loop(self):
        logger.info("Real-time traffic simulation thread started.")
        while self.is_running:
            user_id = random.randint(0, NUM_USERS - 1)
            video_id = random.randint(1, NUM_VIDEOS - 1)
            self.redis_client.add_to_user_history(user_id, video_id)
            if self.click_callback:
                try:
                    self.click_callback(user_id, video_id)
                except Exception as e:
                    logger.error(f"Error executing click simulator callback: {e}")
            sleep_time = random.uniform(0.5, 2.0)
            time.sleep(sleep_time)
        logger.info("Real-time traffic simulation thread stopped.")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)


# =====================================================================
# DRY-RUN SIMULATION BLOCK
# =====================================================================
if __name__ == "__main__":
    print("\033[95m====================================================================\033[0m")
    print("\033[95m🚀 RecoStream Real-Time Event Streaming Simulator (Apache Kafka + Flink)\033[0m")
    print("\033[95m====================================================================\033[0m")
    
    # Initialize Kafka queue and Flink processor
    queue = EventQueue()
    redis_mock = MockRedisClient()
    processor = StreamProcessor(queue, redis_mock)
    
    processor.start()
    
    categories = ["Music", "Gaming", "Tech", "Education", "Vlogs", "News", "Sports", "Comedy"]
    
    # 1. Produce 10,000 mock events rapidly to measure Flink throughput
    logger.info("Producing 10,000 streaming events into Kafka queue...")
    
    t_start = time.time()
    for idx in range(10000):
        uid = random.randint(0, 99)
        vid = random.randint(1, 500)
        cat = categories[vid % len(categories)]
        event = StreamEvent(
            user_id=uid,
            item_id=vid,
            event_type="watch_complete" if random.random() > 0.3 else "click",
            timestamp=time.time(),
            session_id=f"sess_{uid}",
            watch_percentage=random.uniform(20.0, 100.0),
            context={"category": cat}
        )
        queue.produce(event)
        
        # Print flow indicator for the first 5 events
        if idx < 5:
            event_color = "\033[92m" if event.event_type == "watch_complete" else "\033[94m"
            print(f"{event_color}[Kafka Event Produced] User {uid} -> Video {vid} ({cat}) | Type: {event.event_type}\033[0m")
            
    # Wait for Flink stream processor to consume all events
    logger.info("Waiting for Flink Processor background consumer to clear queue...")
    while queue.stats()["queue_size"] > 0:
        time.sleep(0.05)
        
    t_end = time.time()
    elapsed = t_end - t_start
    
    # Calculate stats
    stats = queue.stats()
    latency = (elapsed / 10000) * 1000  # Latency per event in ms
    throughput = 10000 / elapsed
    
    print("\033[93m====================================================================\033[0m")
    print("\033[93m📊 REAL-TIME STREAMING ANALYTICS & LOGISTICS\033[0m")
    print("\033[93m====================================================================\033[0m")
    print(f"✅ Total Events Processed: \033[92m10,000\033[0m")
    print(f"✅ Processing Latency: \033[92m{latency:.4f} ms per event\033[0m (Benchmark Target: < 5ms)")
    print(f"✅ Event Throughput: \033[92m{throughput:.2f} events/second\033[0m")
    print(f"✅ Kafka Queue Remaining: \033[92m{stats['queue_size']}\033[0m (Lag: 0)")
    
    # Display real-time trending items
    trending = processor.get_trending_items(top_k=5)
    print("\033[96m🔥 TOP 5 REAL-TIME TRENDING ITEMS (Last 5 Minutes):\033[0m")
    for rank, (item, views) in enumerate(trending):
        cat = processor.item_categories.get(item, "General")
        print(f"  {rank+1}. Video ID: \033[92m{item}\033[0m | Views: \033[92m{views}\033[0m | Category: {cat}")
        
    # Show user recommendation evolution as mood changes
    test_user = 7
    print(f"\n\033[95m🔄 DEMONSTRATING RECOMMENDATION EVOLUTION FOR USER {test_user}:\033[0m")
    
    recommender = SessionAwareRecommender(None, processor)
    
    # Step A: User starts fresh with no session mood
    res1 = recommender.recommend(test_user, top_n=3)
    print(f"  [Time 0] User mood: \033[94m{res1['current_mood']}\033[0m | Session length: {res1['session_length']}")
    print(f"  Recs: {res1['recommendations']}")
    
    # Step B: User watches 5 gaming videos in a row
    for _ in range(5):
        vid = 104 # gaming index
        processor.item_categories[vid] = "Gaming"
        event = StreamEvent(
            user_id=test_user,
            item_id=vid,
            event_type="watch_complete",
            timestamp=time.time(),
            session_id=f"sess_{test_user}",
            watch_percentage=98.0,
            context={"category": "Gaming"}
        )
        queue.produce(event)
        
    # Wait for processing
    while queue.stats()["queue_size"] > 0:
        time.sleep(0.01)
        
    res2 = recommender.recommend(test_user, top_n=3)
    print(f"  [Time 1] User watches Gaming videos -> Mood shifts to: \033[92m{res2['current_mood']}\033[0m | Session length: {res2['session_length']}")
    print(f"  Recs: {res2['recommendations']}")
    
    # Step C: User switches to watching Education videos
    for _ in range(8):
        vid = 205 # Education index
        processor.item_categories[vid] = "Education"
        event = StreamEvent(
            user_id=test_user,
            item_id=vid,
            event_type="watch_complete",
            timestamp=time.time(),
            session_id=f"sess_{test_user}",
            watch_percentage=95.0,
            context={"category": "Education"}
        )
        queue.produce(event)
        
    while queue.stats()["queue_size"] > 0:
        time.sleep(0.01)
        
    res3 = recommender.recommend(test_user, top_n=3)
    print(f"  [Time 2] User switches to Education -> Mood shifts to: \033[92m{res3['current_mood']}\033[0m | Session length: {res3['session_length']}")
    print(f"  Recs: {res3['recommendations']}")
    
    processor.stop()
