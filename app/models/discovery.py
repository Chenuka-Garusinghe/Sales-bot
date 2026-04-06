"""
Data models for Stage 2: Discovery Call Assistant.

These models represent call transcripts, RAG retrieval matches,
and the AI-generated insights produced by analyzing a sales call.
"""

from pydantic import BaseModel
from datetime import datetime


class TranscriptSegment(BaseModel):
    """A single turn in a sales call conversation."""
    speaker: str          # "rep" or "prospect"
    timestamp_s: float    # seconds from start of call
    text: str             # what was said


class CallTranscript(BaseModel):
    """A complete discovery call transcript with all dialogue turns.
    Used as input to the analysis pipeline.
    """
    call_id: str                        # e.g. "call-001"
    rep_name: str                       # the sales rep
    prospect_name: str                  # the potential customer
    prospect_company: str
    date: datetime
    duration_s: int                     # total call length in seconds
    segments: list[TranscriptSegment]   # the full conversation


class CallTranscriptSummary(BaseModel):
    """Lightweight version of CallTranscript for listing endpoints.
    Omits the full segments to keep the response small.
    """
    call_id: str
    rep_name: str
    prospect_name: str
    prospect_company: str
    date: datetime
    duration_s: int
    segment_count: int   # how many dialogue turns


class RAGMatch(BaseModel):
    """A past meeting retrieved by the mock RAG pipeline (rag.py)
    that is similar to the current call being analyzed.

    In production this would use vector embeddings + cosine similarity.
    In the demo it uses keyword overlap as a proxy.
    """
    matched_call_id: str       # which historical call matched
    similarity_score: float    # 0.0-1.0, higher = more similar
    matched_excerpt: str       # first 200 chars of the historical call's summary
    outcome: str               # "closed_won" or "closed_lost"
    key_technique: str         # what worked (or didn't) in that past call


class CallInsight(BaseModel):
    """AI-generated insights from analyzing a discovery call.
    These are produced by keyword matching (mock) rather than a real LLM.
    """
    deal_risks: list[str]              # e.g. "Prospect mentioned 'budget'"
    objection_patterns: list[str]      # e.g. "Objection pattern: 'price'"
    coaching_tips: list[str]           # actionable advice for the rep
    sentiment_trend: str               # "positive", "neutral", or "declining"
    next_steps_suggested: list[str]    # recommended follow-up actions


class DiscoveryAnalysisResponse(BaseModel):
    """The full response from POST /api/discovery/analyze/{call_id}.
    Bundles the transcript, RAG matches, and generated insights together.
    """
    call_id: str
    transcript: CallTranscript
    rag_matches: list[RAGMatch]
    insights: CallInsight
