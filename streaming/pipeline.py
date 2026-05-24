"""
Real-Time Streaming Data Pipeline Simulation (Apache Kafka + Apache Flink).

Provides:
1. Thread-safe EventQueue (simulating high-velocity Kafka broker streams).
2. Background daemon StreamProcessor (simulating Flink real-time sliding windows).
3. Session-based user aggregates, view counts, and category mood detection.
4. Real-time updates pushed back to the local Feature Store.
5. Legacy StreamingDataPipeline singleton interface for backward compatibility.
"""

import time
import threading
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from loguru import logger
from config import settings

# ==========================================
# 🚀 Modern Real-Time Streaming Components
# ==========================================

@dataclass
class StreamEvent:
    """Represents a structured user interaction event ingested in the streaming pipeline."""
    user_id: int
    item_id: int
    event_type: str
    timestamp: float = field(default_factory=time.time)
    session_id: str = "default"
    watch_percentage: float = 0.0
    category_id: int = 0

class EventQueue:
    """
    Thread-safe circular event buffer simulating an Apache Kafka topic stream.
    """
    def __init__(self, maxlen: int = 100000):
        self._q = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.produced = 0
        self.consumed = 0

    def produce(self, event: StreamEvent):
        """Appends a new interaction event to the queue in a thread-safe manner."""
        with self._lock:
            self._q.append(event)
            self.produced += 1

    def consume_batch(self, n: int = 50) -> List[StreamEvent]:
        """Pops and returns up to n events from the left of the queue in a thread-safe manner."""
        with self._lock:
            batch = []
            for _ in range(min(n, len(self._q))):
                batch.append(self._q.popleft())
            self.consumed += len(batch)
            return batch

    @property
    def lag(self) -> int:
        """Calculates current unprocessed queue lag (Produced - Consumed)."""
        return self.produced - self.consumed

    @property
    def size(self) -> int:
        """Returns the current size of the queue buffer."""
        return len(self._q)

class StreamProcessor:
    """
    Simulates an Apache Flink stream processor operating on rolling windows in a background thread.
    Accumulates real-time statistics, sliding watch history, and moods, pushing updates to the Feature Store.
    """
    def __init__(self, queue: EventQueue, feature_store: Optional[Any] = None):
        self.queue = queue
        self.feature_store = feature_store
        self.running = False
        
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Thread-safe aggregate structures
        self.user_sessions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        self.item_views: Dict[int, int] = defaultdict(int)
        
        self.processed = 0
        self.errors = 0
        self._window = 300  # Rolling sliding window (5 minutes in seconds)

    def start(self):
        """Spins up the background processing thread as a daemon."""
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Stream processor started ⚡")

    def stop(self):
        """Gracefully terminates background processing loop."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info(f"Stream processor stopped. Total processed events: {self.processed}")

    def _loop(self):
        """Continuous ingestion loop pulling microbatches from the event queue."""
        while self.running:
            batch = self.queue.consume_batch(50)
            if not batch:
                time.sleep(0.01)
                continue

            for event in batch:
                try:
                    self._handle(event)
                    with self._lock:
                        self.processed += 1
                except Exception as e:
                    with self._lock:
                        self.errors += 1
                    logger.error(f"Error handling streaming event: {e}")

    def _handle(self, event: StreamEvent):
        """Updates internal rolling sliding windows and caches results in Feature Store."""
        now = time.time()
        
        with self._lock:
            # 1. Append session event
            self.user_sessions[event.user_id].append({
                "item_id": event.item_id,
                "category_id": event.category_id,
                "ts": event.timestamp,
                "event_type": event.event_type
            })
            
            # 2. Evict expired entries older than sliding window limit (5 mins)
            cutoff = now - self._window
            self.user_sessions[event.user_id] = deque(
                [entry for entry in self.user_sessions[event.user_id] if entry["ts"] >= cutoff],
                maxlen=50
            )

            # 3. Increment item view counts
            if event.event_type in ("view", "click", "watch_complete"):
                self.item_views[event.item_id] += 1

        # 4. Push updates to Feature Store in real-time
        if self.feature_store:
            with self._lock:
                recent_items = [entry["item_id"] for entry in list(self.user_sessions[event.user_id])[-10:]]
                session_len = len(self.user_sessions[event.user_id])

            # Perform storage update outside lock to avoid DB serialization delays
            user_feat = self.feature_store.get_user(event.user_id) or {}
            user_feat["recent_items"] = recent_items
            user_feat["last_event"] = event.event_type
            user_feat["session_length"] = session_len
            self.feature_store.store_user(event.user_id, user_feat)

            # Re-sync overall watch history list in local Feature Store
            with self._lock:
                all_items = [entry["item_id"] for entry in list(self.user_sessions[event.user_id])]
            self.feature_store.store_history(event.user_id, all_items)

    def get_trending(self, top_k: int = 20) -> List[tuple]:
        """Returns top_k items by view count in the active window."""
        with self._lock:
            sorted_views = sorted(self.item_views.items(), key=lambda x: x[1], reverse=True)
            return sorted_views[:top_k]

    def get_session(self, user_id: int) -> List[dict]:
        """Retrieves user's rolling watch click session log."""
        with self._lock:
            return list(self.user_sessions[user_id])

    def get_mood(self, user_id: int) -> Optional[int]:
        """Calculates dynamic taste mood defined as the most frequent category_id in the last 10 session entries."""
        with self._lock:
            entries = list(self.user_sessions[user_id])[-10:]
            
        if not entries:
            return None

        counts = defaultdict(int)
        for entry in entries:
            counts[entry["category_id"]] += 1

        sorted_cats = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_cats[0][0]

    def stats(self) -> dict:
        """Returns active aggregate processing telemetry."""
        with self._lock:
            active_users = sum(1 for s in self.user_sessions.values() if len(s) > 0)
            trending_count = len(self.item_views)
            proc = self.processed
            err = self.errors

        return {
            "processed": proc,
            "errors": err,
            "active_users": active_users,
            "trending_items": trending_count,
            "queue_lag": self.queue.lag
        }


# ==========================================
# 🔄 Legacy / UI Compatibility Layer
# ==========================================

class StreamingDataPipeline:
    """
    Legacy class supporting micro-batch processes under RecoStream.
    Ensures that existing tests and Gradio UI dashboard continue to work.
    """
    def __init__(self):
        self.event_queue = deque(maxlen=1000)
        self.processed_count = 0
        self.window_clicks = 0
        self.window_watch_time = 0.0

    def ingest_interaction_event(self, user_id: int, video_id: int, click: int, watch_ratio: float):
        """Simulates Kafka event broker ingestion."""
        event = {
            "user_id": user_id,
            "video_id": video_id,
            "click": click,
            "watch_ratio": watch_ratio,
            "timestamp": time.time()
        }
        self.event_queue.append(event)

    def process_next_microbatch(self, batch_size: int = 10) -> list:
        """Simulates Flink micro-batch stream window aggregations."""
        batch_events = []
        for _ in range(min(batch_size, len(self.event_queue))):
            event = self.event_queue.popleft()
            batch_events.append(event)
            
            self.processed_count += 1
            if event["click"] == 1:
                self.window_clicks += 1
                self.window_watch_time += event["watch_ratio"] * 0.1
                
        return batch_events

    def get_window_telemetry(self) -> dict:
        """Retrieves active streaming metrics compiled over the sliding window."""
        return {
            "processed_events": self.processed_count,
            "active_window_clicks": self.window_clicks,
            "active_window_watch_hours": float(round(self.window_watch_time, 2)),
            "pipeline_status": "ONLINE (Processing real-time stream)"
        }

# Instantiate pipeline singleton
streaming_pipeline = StreamingDataPipeline()
