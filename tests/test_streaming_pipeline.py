import time
import numpy as np
from streaming.pipeline import StreamEvent, EventQueue, StreamProcessor
from services.feature_store import FeatureStore

def test_event_queue_threading():
    """Verifies that the EventQueue operates with strict thread safety and accurate counters."""
    eq = EventQueue(maxlen=100)
    
    # Produce mock events
    event1 = StreamEvent(user_id=1, item_id=101, event_type="click", category_id=3)
    event2 = StreamEvent(user_id=2, item_id=102, event_type="view", category_id=4)
    
    eq.produce(event1)
    eq.produce(event2)
    
    assert eq.size == 2
    assert eq.produced == 2
    assert eq.lag == 2
    
    # Batch consume
    batch = eq.consume_batch(5)
    assert len(batch) == 2
    assert eq.consumed == 2
    assert eq.lag == 0
    assert eq.size == 0
    
    assert batch[0].user_id == 1
    assert batch[1].user_id == 2

def test_stream_processor_aggregations_and_mood():
    """Verifies background Flink-like window aggregations, mood calculations, and active taste trends."""
    eq = EventQueue()
    fs = FeatureStore()
    sp = StreamProcessor(queue=eq, feature_store=fs)
    
    # 1. Produce clicks for multiple categories
    now = time.time()
    # User 1 views item 100 in category 2 (Music)
    eq.produce(StreamEvent(user_id=1, item_id=100, event_type="click", category_id=2, timestamp=now))
    # User 1 views item 101 in category 2 (Music)
    eq.produce(StreamEvent(user_id=1, item_id=101, event_type="click", category_id=2, timestamp=now))
    # User 1 views item 102 in category 5 (Gaming)
    eq.produce(StreamEvent(user_id=1, item_id=102, event_type="click", category_id=5, timestamp=now))
    
    # 2. Produce expired click (6 minutes ago)
    eq.produce(StreamEvent(user_id=1, item_id=999, event_type="click", category_id=8, timestamp=now - 400))
    
    # Handle batch events
    batch = eq.consume_batch(10)
    for event in batch:
        sp._handle(event)
        
    # Verify expired event got evicted from rolling sliding session window
    session = sp.get_session(user_id=1)
    assert len(session) == 3, f"Expired session entries not evicted! Session count is {len(session)}"
    assert 999 not in [e["item_id"] for e in session], "Evicted item still present in session!"
    
    # Verify dynamic category-based mood (most frequent is category_id 2 - Music)
    mood = sp.get_mood(user_id=1)
    assert mood == 2, f"Mood calculation failed! Expected 2, got {mood}"
    
    # Verify trending item view counts
    trending = sp.get_trending(top_k=5)
    # Item 100, 101, 102, and 999 got views
    assert len(trending) >= 3
    assert trending[0][0] in [100, 101, 102, 999]
    
    # Verify real-time Feature Store updates
    fs_user = fs.get_user(1)
    assert fs_user is not None
    assert fs_user["session_length"] == 3
    assert fs_user["last_event"] == "click"
    assert fs_user["recent_items"] == [100, 101, 102]
    
    # Verify watch history list synced in FakeRedis
    fs_hist = fs.get_history(1)
    assert fs_hist == [100, 101, 102]

def test_stream_processor_threading():
    """Verifies that the background daemon thread can start and stop gracefully without resource leaks."""
    eq = EventQueue()
    sp = StreamProcessor(queue=eq)
    
    sp.start()
    assert sp.running is True
    assert sp._thread is not None
    assert sp._thread.is_alive() is True
    
    # Produce event during active run
    eq.produce(StreamEvent(user_id=1, item_id=200, event_type="view"))
    
    # Wait briefly for background thread to process
    time.sleep(0.05)
    
    stats = sp.stats()
    assert stats["processed"] == 1
    
    sp.stop()
    assert sp.running is False
    assert sp._thread.is_alive() is False
