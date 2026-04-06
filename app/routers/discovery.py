"""
Router for Stage 2: Discovery Call Assistant endpoints.

All endpoints are prefixed with /api/discovery (set in main.py).

Endpoints:
    GET  /api/discovery/transcripts            -- list all available call transcripts
    GET  /api/discovery/transcripts/{call_id}   -- get a full transcript
    POST /api/discovery/analyze/{call_id}       -- analyze a call (RAG + insights)
    GET  /api/discovery/datastore               -- view the growing past meetings store
"""

from fastapi import APIRouter, HTTPException

from app.models.discovery import CallTranscript, CallTranscriptSummary, DiscoveryAnalysisResponse
from app.services.discovery_call import analyze_call, get_datastore, get_transcript, list_transcripts

router = APIRouter()


@router.get("/transcripts", response_model=list[CallTranscriptSummary])
async def get_transcripts():
    """List all available mock transcripts (summary view without full dialogue).
    Available call IDs: call-001, call-002, call-003.
    """
    return list_transcripts()


@router.get("/transcripts/{call_id}", response_model=CallTranscript)
async def get_transcript_detail(call_id: str):
    """Get a full transcript including all dialogue segments.
    Use this to read the raw call before or after analysis.
    """
    transcript = get_transcript(call_id)
    if not transcript:
        raise HTTPException(status_code=404, detail=f"Transcript '{call_id}' not found")
    return transcript


@router.post("/analyze/{call_id}", response_model=DiscoveryAnalysisResponse)
async def analyze_discovery_call(call_id: str):
    """Analyze a discovery call through the full pipeline:
    1. Load transcript
    2. RAG retrieval (find similar past meetings)
    3. Extract deal risks, objection patterns, sentiment
    4. Generate coaching tips
    5. Add to growing datastore

    Each call to this endpoint grows the datastore by one entry.
    """
    result = analyze_call(call_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Transcript '{call_id}' not found")
    return result


@router.get("/datastore")
async def view_datastore():
    """View the growing datastore of past meetings.
    Starts with 8 historical entries. Grows by 1 each time /analyze is called.
    This demonstrates the "growing knowledge base" concept from the design doc.
    """
    store = get_datastore()
    return {"total_entries": len(store), "entries": store}
