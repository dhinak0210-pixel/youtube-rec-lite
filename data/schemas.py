"""
Pydantic v2 Data Schemas for YouTube-scale Recommendation System.

Declares all core validation schemas representing users, catalog items, 
dynamic interaction events, ranking boundaries, and API request/response payloads.
Includes older schemas for full pipeline backward compatibility.
"""

import time
from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# ==========================================
# 🚀 Pydantic v2 Requested Schemas
# ==========================================

class InteractionType(str, Enum):
    """Enumeration of active user feedback actions."""
    VIEW = "VIEW"
    CLICK = "CLICK"
    LIKE = "LIKE"
    DISLIKE = "DISLIKE"
    SHARE = "SHARE"
    WATCH_COMPLETE = "WATCH_COMPLETE"

# Mapping weights of different interactions to rank interest levels
INTERACTION_WEIGHTS: Dict[InteractionType, float] = {
    InteractionType.VIEW: 1.0,
    InteractionType.CLICK: 2.0,
    InteractionType.LIKE: 3.0,
    InteractionType.SHARE: 4.0,
    InteractionType.WATCH_COMPLETE: 5.0,
    InteractionType.DISLIKE: -2.0
}

class UserProfile(BaseModel):
    """Strict representation of a user profile with categorical demographics."""
    user_id: int
    age_bucket: int = Field(..., ge=0, le=6, description="Age bracket index between 0 and 6.")
    gender: int = Field(..., ge=0, le=2, description="Gender mapping integer index (0-2).")
    country_id: int = Field(..., ge=0, le=99, description="Country classification integer index (0-99).")
    signup_days_ago: int
    num_interactions: int = Field(default=0, description="Total user engagement click count.")
    preferred_categories: List[int] = Field(default_factory=list, description="Categorical vertical IDs.")

    @property
    def is_cold_start(self) -> bool:
        """Determines if user interaction logs are insufficient."""
        return self.num_interactions < 5

class ItemProfile(BaseModel):
    """Strict representation of video catalog item attributes and views."""
    item_id: int
    category_id: int
    duration_seconds: int
    upload_days_ago: int
    creator_id: int
    total_views: int = Field(default=0)
    like_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    avg_watch_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    content_embedding: List[float] = Field(default_factory=list)

class Interaction(BaseModel):
    """High-fidelity log of single User-Item interaction event."""
    user_id: int
    item_id: int
    interaction_type: InteractionType
    timestamp: float = Field(default_factory=time.time)
    watch_percentage: float = Field(default=0.0, ge=0.0, le=1.0)
    context_hour: int = Field(default=12, ge=0, le=23)
    context_device: int = Field(default=0, ge=0, le=3)

    @property
    def weight(self) -> float:
        """Calculates interaction importance scalar score."""
        base_w = INTERACTION_WEIGHTS.get(self.interaction_type, 0.0)
        if self.interaction_type == InteractionType.VIEW:
            return base_w * self.watch_percentage
        return base_w

class ScoredItem(BaseModel):
    """Scored candidate item retrieved from a pipeline source."""
    item_id: int
    score: float
    source: str = Field(default="unknown")

class RecommendationRequest(BaseModel):
    """Personalized API candidate generation request schema."""
    user_id: int
    num_recommendations: int = Field(default=20, le=100)
    exclude_items: List[int] = Field(default_factory=list)
    context_hour: int = Field(default=12, ge=0, le=23)
    context_device: int = Field(default=0, ge=0, le=3)

class RecommendationResponse(BaseModel):
    """Unified candidate recommendation response schema."""
    user_id: int
    recommendations: List[ScoredItem]
    model_version: str
    latency_ms: float
    is_cold_start: bool = Field(default=False)
    ab_test_group: Optional[str] = None
    sources_breakdown: Dict[str, int] = Field(default_factory=dict)
    explanation: List[str] = Field(default_factory=list)

class FeedbackEvent(BaseModel):
    """Log schema to capture immediate post-recommendation user feedback."""
    user_id: int
    item_id: int
    interaction_type: InteractionType
    timestamp: float = Field(default_factory=time.time)
    ab_test_group: Optional[str] = None


# ==========================================
# 🔄 Backward Compatibility Legacy Schemas
# ==========================================

class UserSchema(BaseModel):
    """Legacy user configuration validation format."""
    user_id: int = Field(..., description="Unique integer ID of the user.")
    age: int = Field(..., ge=13, le=100, description="Age of the user.")
    gender: str = Field(..., pattern="^(M|F|O)$", description="Gender of the user (M/F/O).")
    country: str = Field(..., min_length=2, max_length=3, description="ISO country code.")
    preferred_categories: List[str] = Field(default_factory=list, description="Preferred video categories.")
    signup_date: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp.")

class VideoSchema(BaseModel):
    """Legacy item configuration validation format."""
    video_id: int = Field(..., description="Unique integer ID of the video.")
    category: str = Field(..., description="Primary vertical/category of the video.")
    tags: List[str] = Field(default_factory=list, description="Metadata tags for semantic matching.")
    duration_seconds: int = Field(..., gt=0, description="Duration in seconds.")
    upload_date: datetime = Field(default_factory=datetime.utcnow, description="Video upload timestamp.")
    views_count: int = Field(default=0, ge=0, description="Cumulative watch view count.")
    likes_count: int = Field(default=0, ge=0, description="Cumulative positive reviews count.")

class InteractionSchema(BaseModel):
    """Legacy user click interaction tracking format."""
    user_id: int = Field(..., description="ID of the interacting user.")
    video_id: int = Field(..., description="ID of the targeted video.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Interaction logging timestamp.")
    click: int = Field(..., ge=0, le=1, description="Binary flag indicating click (1) or impression (0).")
    watch_ratio: float = Field(..., ge=0.0, le=1.0, description="Watched duration percentage (0.0 to 1.0).")
    like: int = Field(..., ge=0, le=1, description="Positive sentiment action.")
    dislike: int = Field(..., ge=0, le=1, description="Negative sentiment action.")
    share: int = Field(..., ge=0, le=1, description="Virality sharing action.")
