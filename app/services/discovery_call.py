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

RISK_KEYWORDS = {"budget", "timeline", "competitor", "unsure", "expensive", "skeptical", "worried", "risk"}
OBJECTION_KEYWORDS = {"cost", "price", "cheaper", "alternative", "not sure", "too much", "pushback"}

# Growing datastore - starts with PAST_MEETINGS, grows as calls are analyzed
_datastore: list[dict] = list(PAST_MEETINGS)


def list_transcripts() -> list[CallTranscriptSummary]:
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
    transcript = get_transcript(call_id)
    if not transcript:
        return None

    full_text = " ".join(seg.text for seg in transcript.segments)

    rag_matches = retrieve_similar_meetings(full_text)

    words = set(full_text.lower().split())
    risks = [f"Prospect mentioned '{kw}'" for kw in RISK_KEYWORDS if kw in words]
    objections = [f"Objection pattern: '{kw}'" for kw in OBJECTION_KEYWORDS if kw in words]

    positive = len(words & {"great", "interested", "excited", "definitely", "love", "fantastic", "perfect", "excellent"})
    negative = len(words & {"concerned", "worried", "expensive", "unsure", "difficult", "skeptical", "pushy"})
    sentiment = "positive" if positive > negative else ("declining" if negative > positive else "neutral")

    coaching: list[str] = []
    if rag_matches:
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

    # Add to growing datastore
    call_keywords = [w for w in words if len(w) > 4][:10]
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
    return _datastore
