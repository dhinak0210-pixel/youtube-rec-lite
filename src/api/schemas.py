from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="Target user index for personalization retrieval")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of sorted video recommendations to return")

class VideoRecommendation(BaseModel):
    video_id: int = Field(..., description="Unique Video identifier")
    score: float = Field(..., description="Calculated matching relevance rank score")
    category: str = Field(..., description="Categorical class identifier of the video")
    duration: int = Field(..., description="Video duration in seconds")

class RecommendationResponse(BaseModel):
    user_id: int
    group: str = Field(..., description="Assigned user experiment group (Control or Treatment)")
    recommendations: List[VideoRecommendation] = Field(..., description="List of ranked candidates")
    cached: bool = Field(default=False, description="Indicates if serving cached predictions")
    sources_breakdown: Optional[Dict[str, int]] = Field(default=None, description="Candidate count breakdown per source")
    latency_breakdown: Optional[Dict[str, float]] = Field(default=None, description="Time taken per processing stage in ms")
    explanations: Optional[Dict[int, str]] = Field(default=None, description="Human-readable recommendations explanations")
    is_cold_start: Optional[bool] = Field(default=False, description="Indicates cold start user")


class InteractionEvent(BaseModel):
    user_id: int
    video_id: int
    click: int = Field(..., ge=0, le=1)
    watch_ratio: float = Field(..., ge=0.0, le=1.0)
    like: int = Field(default=0, ge=0, le=1)

class ExperimentMetrics(BaseModel):
    group: str
    impressions: int
    clicks: int
    ctr: float
    avg_watch_ratio: float
    like_ratio: float

class DashboardReport(BaseModel):
    experiment_name: str
    metrics: Dict[str, ExperimentMetrics]
