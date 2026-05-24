import json
import fakeredis
from typing import List, Dict, Optional
from src.utils.logger import logger

class MockRedisClient:
    """
    In-memory mock Redis cache manager leveraging 'fakeredis'.
    Operates as the real-time user state and historical interaction logger.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MockRedisClient, cls).__new__(cls, *args, **kwargs)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        logger.info("Initializing in-memory fakeredis client.")
        self.r = fakeredis.FakeStrictRedis()

    def get_user_history(self, user_id: int) -> List[int]:
        """
        Retrieves real-time session click history sequence for a user.
        """
        key = f"user_history:{user_id}"
        history_bytes = self.r.lrange(key, 0, -1)
        if not history_bytes:
            return []
        return [int(val.decode("utf-8")) for val in history_bytes]

    def add_to_user_history(self, user_id: int, video_id: int, max_len: int = 15):
        """
        Pushes a new clicked video to the user's real-time action queue.
        Caps length at max_len to represent dynamic sliding history.
        """
        key = f"user_history:{user_id}"
        self.r.rpush(key, video_id)
        # Trim list from the left if it exceeds max_len
        self.r.ltrim(key, -max_len, -1)
        logger.debug(f"Added video {video_id} to real-time history for user {user_id}.")

    def set_cached_recommendations(self, user_id: int, recommendations: List[int], ttl: int = 3600):
        """
        Caches final ranked recommendations to bypass model evaluation for repeated loads.
        """
        key = f"user_recs:{user_id}"
        data = json.dumps(recommendations)
        self.r.setex(key, ttl, data)

    def get_cached_recommendations(self, user_id: int) -> Optional[List[int]]:
        """
        Returns cached recommendations if present, otherwise returns None.
        """
        key = f"user_recs:{user_id}"
        data = self.r.get(key)
        if data:
            logger.debug(f"Cache hit! Serving pre-calculated recs for user {user_id}.")
            return json.loads(data.decode("utf-8"))
        return None

    def clear_cache(self):
        """
        Clears all in-memory database keys.
        """
        self.r.flushall()
        logger.info("Fakeredis cache flushed.")
