"""
Stage 2: Discovery Call Analysis Service

This service processes sales call transcripts and generates insights.
It's the orchestrator for the discovery call pipeline:

    1. Load the transcript from mock data
    2. Run RAG retrieval to find similar past meetings
    3. Extract deal risks and objection patterns via keyword matching
    4. Compute sentiment (positive vs negative word counts)
    5. Generate coaching tips based on RAG matches and detected patterns
    6. Add the analyzed call to the growing datastore (so future RAG searches can find it)

The "growing datastore" concept: every time you analyze a call, it gets
added to _datastore. This means the RAG corpus gets richer over time,
simulating a learning system. (In production, you'd store these in a vector DB.)
"""

from app.mock_data.past_meetings import PAST_MEETINGS
from app.mock_data.transcripts import TRANSCRIPTS
from app.models.discovery import (
    CallInsight,
    CallTranscript,
    CallTranscriptSummary,
    DiscoveryAnalysisResponse,
    TranscriptSegment,
)
from app.services.rag import retrieve_similar_meetings

# Keywords that signal potential deal risks when mentioned by the prospect
RISK_KEYWORDS = {"budget", "timeline", "competitor", "unsure", "expensive", "skeptical", "worried", "risk"}

# Keywords that signal objection patterns
OBJECTION_KEYWORDS = {"cost", "price", "cheaper", "alternative", "not sure", "too much", "pushback"}

# Growing datastore -- starts with historical meetings, grows as calls are analyzed.
# This is in-memory only, resets when the server restarts.
_datastore: list[dict] = list(PAST_MEETINGS)


def list_transcripts() -> list[CallTranscriptSummary]:
    """Return a lightweight summary of all available transcripts.
    Used by GET /api/discovery/transcripts.
    """
    summaries = []
    for data in TRANSCRIPTS.values():
        summaries.append(
            CallTranscriptSummary(
                call_id=data["call_id"],
                rep_name=data["rep_name"],
                prospect_name=data["prospect_name"],
                prospect_company=data["prospect_company"],
                date=data["date"],
                duration_s=data["duration_s"],
                segment_count=len(data["segments"]),
            )
        )
    return summaries


def get_transcript(call_id: str) -> CallTranscript | None:
    """Load a full transcript by ID. Returns None if not found."""
    data = TRANSCRIPTS.get(call_id)
    if not data:
        return None
    return CallTranscript(
        call_id=data["call_id"],
        rep_name=data["rep_name"],
        prospect_name=data["prospect_name"],
        prospect_company=data["prospect_company"],
        date=data["date"],
        duration_s=data["duration_s"],
        segments=[TranscriptSegment(**seg) for seg in data["segments"]],
    )


def analyze_call(call_id: str) -> DiscoveryAnalysisResponse | None:
    """Run the full analysis pipeline on a call transcript.

    This is the main function called by POST /api/discovery/analyze/{call_id}.

    Returns None if the call_id doesn't exist in mock data.
    """
    transcript = get_transcript(call_id)
    if not transcript:
        return None

    # Concatenate all dialogue into a single string for analysis
    full_text = " ".join(seg.text for seg in transcript.segments)

    # Step 1: RAG retrieval -- find similar past meetings
    rag_matches = retrieve_similar_meetings(full_text)

    # Step 2: Extract deal risks and objections via keyword matching
    words = set(full_text.lower().split())
    risks = [f"Prospect mentioned '{kw}'" for kw in RISK_KEYWORDS if kw in words]
    objections = [f"Objection pattern: '{kw}'" for kw in OBJECTION_KEYWORDS if kw in words]

    # Step 3: Compute sentiment by counting positive vs negative words
    positive = len(words & {"great", "interested", "excited", "definitely", "love", "fantastic", "perfect", "excellent"})
    negative = len(words & {"concerned", "worried", "expensive", "unsure", "difficult", "skeptical", "pushy"})
    sentiment = "positive" if positive > negative else ("declining" if negative > positive else "neutral")

    # Step 4: Generate coaching tips
    coaching: list[str] = []
    if rag_matches:
        # Reference what worked in a similar past call
        best = rag_matches[0]
        coaching.append(
            f"Similar call ({best.matched_call_id}) used technique "
            f"'{best.key_technique}' and resulted in {best.outcome}"
        )
    if objections:
        coaching.append("Consider addressing price objections earlier with ROI framing")
    if "budget" in words and "approved" not in words:
        coaching.append("Budget not confirmed - qualify spend authority in next interaction")
    coaching.append("Ask more open-ended questions about their current workflow pain points")

    insights = CallInsight(
        deal_risks=risks or ["No significant risks detected"],
        objection_patterns=objections or ["No strong objection patterns"],
        coaching_tips=coaching,
        sentiment_trend=sentiment,
        next_steps_suggested=[
            "Schedule follow-up within 48 hours",
            "Send case study relevant to their industry",
            "Identify and engage additional stakeholders",
        ],
    )

    # Step 5: Add to growing datastore so future RAG queries can find this call
    call_keywords = [w for w in words if len(w) > 4][:10]  # top 10 words with 5+ chars
    _datastore.append({
        "call_id": call_id,
        "summary": f"Discovery call with {transcript.prospect_name} at {transcript.prospect_company}. Sentiment: {sentiment}.",
        "outcome": "in_progress",
        "key_technique": coaching[0] if coaching else "standard discovery",
        "keywords": call_keywords,
    })

    return DiscoveryAnalysisResponse(
        call_id=call_id,
        transcript=transcript,
        rag_matches=rag_matches,
        insights=insights,
    )


def get_datastore() -> list[dict]:
    """Return the full datastore contents. Used by GET /api/discovery/datastore.
    The datastore starts with 8 historical meetings and grows each time
    analyze_call() is called.
    """
    return _datastore
