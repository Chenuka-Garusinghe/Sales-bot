from pydantic import BaseModel
from datetime import datetime


class TranscriptSegment(BaseModel):
    speaker: str
    timestamp_s: float
    text: str


class CallTranscript(BaseModel):
    call_id: str
    rep_name: str
    prospect_name: str
    prospect_company: str
    date: datetime
    duration_s: int
    segments: list[TranscriptSegment]


class CallTranscriptSummary(BaseModel):
    call_id: str
    rep_name: str
    prospect_name: str
    prospect_company: str
    date: datetime
    duration_s: int
    segment_count: int


class RAGMatch(BaseModel):
    matched_call_id: str
    similarity_score: float
    matched_excerpt: str
    outcome: str
    key_technique: str


class CallInsight(BaseModel):
    deal_risks: list[str]
    objection_patterns: list[str]
    coaching_tips: list[str]
    sentiment_trend: str
    next_steps_suggested: list[str]


class DiscoveryAnalysisResponse(BaseModel):
    call_id: str
    transcript: CallTranscript
    rag_matches: list[RAGMatch]
    insights: CallInsight
